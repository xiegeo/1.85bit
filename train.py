import time, os, json, socket
import torch, wandb
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from collections import deque

from modeling_bitnet import BitnetForCausalLM 
from tokenization_bitnet import BitnetTokenizer 
from transformers import AutoTokenizer

from models import tiny_stories_ref, bitnet_ref, llama_ref
from utils_quant import BitLinear, quantize_weights, QF_noop, QF_3, get_weight_distribution

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
print(f'use {device}')


tokenizer_path = "1bitLLM/bitnet_b1_58-large"
tokenizer = BitnetTokenizer.from_pretrained(tokenizer_path)
#tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

tokenizer.pad_token = tokenizer.eos_token # use eos_token as padding token for opened-ended generation

batch_size = 8 # use the same batch size for consistent reporting
#if device.type == 'cpu':
#    batch_size = min(batch_size, 8)



def get_data_loader(dataset_type,train_subset, max_length, shuffle=True, pre_generate=False):
    sfn = f"tokenized_{dataset_type}_dataset_{max_length}"
    if tokenizer_path == "1bitLLM/bitnet_b1_58-large":
        sfn = f"bitnet_{dataset_type}_tokens_{max_length}"
    if not os.path.exists(sfn) and os.path.exists("../"+sfn): # find the file if it's in the parent directory
        sfn = "../"+sfn
    if os.path.exists(sfn):
        tokenized_dataset = load_from_disk(sfn)
    else:
        # use the TinyStories dataset
        dataset = load_dataset('roneneldan/TinyStories')
        print(dataset.keys())
        if dataset_type not in dataset:
            raise ValueError(f"dataset type {dataset_type} not found in dataset {dataset.keys()}")
        sub_dataset = dataset[dataset_type]
        if not pre_generate:
            sub_dataset = sub_dataset.select(range(train_subset))
        tokenized_dataset = sub_dataset.map(
            lambda x: tokenizer(
                x['text'], padding="max_length", max_length=max_length, truncation=True, return_tensors='pt'
            ), batched=True)
        if pre_generate:
            tokenized_dataset.save_to_disk(sfn)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    print(f"use {dataset_type} dataset with max_length={max_length}, train_subset={train_subset}, number of tokens={max_length*train_subset}")
    tokenized_dataset = torch.utils.data.Subset(tokenized_dataset, indices=range(train_subset))

    # Define your dataloaders
    return torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=shuffle)

def get_training_loader(train_subset, max_length):
    return get_data_loader('train', train_subset, max_length, shuffle=True)

def get_validation_loader(train_subset, max_length):
    return get_data_loader('validation', train_subset, max_length, shuffle=False)


def AdamWFun(lr=1e-3, betas=(0.9, 0.999)):
    def fn(params):
        return torch.optim.AdamW(params, lr=lr, betas=betas)
    fn.summery = f"AdamW(lr={lr}, betas={betas})"
    return fn

def SGDFun(lr=1e-3):
    def fn(params):
        return torch.optim.SGD(params, lr=lr)
    fn.summery = f"SGD(lr={lr})"
    return fn

class DynamicLearningRate(_LRScheduler):
    def __init__(self, optimizer: Optimizer, disable=True, lr_decay=0.8, lr_ratio=0.8, slow_start=1, swap_width=500, swaps=1, lr_min=1e-6):
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.lr_max = self.base_lrs[0] 
        self.disable = disable # always return the base learning rate, but still report states for consistency.
        if disable:
            lr_min = self.lr_max
        self.current_lr = self.lr_max * slow_start
        self.lr_decay = lr_decay # how fast the learning rate adjusts
        self.lr_ratio = lr_ratio # when trying 2 different learning rates, how far apart are they from the current learning rate
        self.swap_width = swap_width
        self.swaps = swaps
        self.decision_steps = 2 * swap_width * swaps
        self.lr_min = lr_min
        self.last_avg = 0
        self.current_avg = 0
        self.deltas = [0,0]
        self.loss_count = 0
        self.loss_weight_scale = 2/((self.swap_width+1)*self.swap_width) # 1/(1+2+3+...+swap_width)
        super(DynamicLearningRate, self).__init__(optimizer)
    
    def higher_decay(self):
        return min(self.current_lr/self.lr_decay, self.lr_max)
    
    def lower_decay(self):
        if self.current_lr == self.lr_min:
            return self.lr_max # if the current learning rate is already the minimum, then cycle to the maximum
        return max(self.current_lr*self.lr_decay, self.lr_min)
    
    def higher_ratio(self):
        return min(self.current_lr/self.lr_ratio, self.lr_max)
    
    def lower_ratio(self):
        return max(self.current_lr*self.lr_ratio, self.lr_min) 

    def lose_index(self):
        return (self.loss_count//self.swap_width +1)%2 # 0 or 1 (+1 when lower ratio goes first)
    
    def lose_weight(self):
        return (self.loss_count%self.swap_width+1)*self.loss_weight_scale
    
    def record_lose(self, loss):
        self.current_avg += loss * self.lose_weight()
        last_index = self.lose_index()
        self.loss_count += 1
        if self.loss_count%self.swap_width == 0:
            old_last_avg = self.last_avg
            self.deltas[last_index] += self.current_avg - self.last_avg
            self.last_avg = self.current_avg
            self.current_avg = 0
            if old_last_avg == 0: # the last average was not set, so we restart this swap cycle with a useful last average
                self.update_lr(self.current_lr)
                return
            
        if self.loss_count == self.decision_steps:
            if self.deltas[1]<self.deltas[0]:
                self.update_lr(self.lower_decay())
            else:
                self.update_lr(self.higher_decay())
            
    def update_lr(self, new_lr):
        #if new_lr != self.current_lr:
        tqdm.write(f'update lr to {new_lr}')
        self.current_lr = new_lr
        self.deltas = [0,0]
        self.loss_count = 0
    
    def get_lr(self):
        if self.disable:
            return self.base_lrs
        if self.lose_index() == 0: 
            return [self.higher_ratio()]
        return [self.lower_ratio()]
    
    def get_states(self):
        return {
            'current_lr':self.current_lr,
            'delta_0':self.deltas[0],
            'delta_1':self.deltas[1],
            'current_avg':self.current_avg,
            'last_avg':self.last_avg,
        }

def train(model,model_name, cost, train_subset = 1024*16, max_length=64, optimizer_function=AdamWFun(), QF=QF_noop):
    
    BitLinear.QW = (QF == QF_3)
    
    # Initialize your model
    #model = AutoModel.from_config(AutoConfig.from_dict(bitnet_64_2))
    model = model.to(device)
    model_save_path = f'model_data/{model_name}_{max_length}/{time.time()}'

    # Prepare the optimizer
    optimizer = optimizer_function(model.parameters())

    loss_fct = torch.nn.CrossEntropyLoss().to(device)
    
    scheduler = DynamicLearningRate(optimizer)

    def calculate_loss(model, input):
        output = model(input,
                        use_cache=False,
                        output_hidden_states=False,
                        output_attentions=False)[0]
        shift_logits = output[:, :-1, :].contiguous()
        shift_labels = input[:, 1:]
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        return loss
    
    validation_size = (10000//batch_size) * batch_size
    if device.type == 'cpu':
        validation_size = (1000//batch_size) * batch_size

    wandb.init(project="npl185", name=model_name,# mode="offline",
            config={
                "tokenizer_path":tokenizer_path,
                "tokenizer_max_length": max_length,
                "train_subset": train_subset,
                "train_batch_size": batch_size, 
                "validation_size": validation_size,
                "model_save_path": model_save_path,
                "model_name": model_name,
                "device": device.type,
                "optimizer": optimizer.__class__.__name__,
                "optimizer_summary": optimizer_function.summery,
                "default_stochastic_rounding": BitLinear.default_stochastic_rounding,
                "quantize_training_weights": QF.__name__,
                })
    validation_loader = None
    if validation_size > 0:
        validation_loader = get_validation_loader(validation_size,max_length)
        
    def sample_output(model, batch_idx=-1, validation_size=0):
        return_to_train = False
        if model.training:
            model.eval()
            return_to_train = True
        with torch.no_grad():
            for text in ["Once","Alice and Bob", "In a galaxy far far away","The lazy dog"]:
                inputs = tokenizer(text, return_tensors='pt').to(device)
                outputs = model.generate(**inputs, max_length=max_length)
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                tqdm.write(json.dumps(decoded))
                wandb.log({'batch_idx':batch_idx, 'text': text, 'story': json.dumps(decoded), "stochastic_rounding": BitLinear.default_stochastic_rounding})
            if validation_size > 0:
                total_loss = 0.0
                batches = 0
                progress_bar = tqdm(validation_loader, position=0)
                for _, batch in enumerate(progress_bar):
                    loss = calculate_loss(model, batch['input_ids'].to(device))
                    total_loss += loss.item()
                    batches += 1
                    if batches * batch_size >= validation_size:
                        break
                    progress_bar.set_postfix({'validation_loss': total_loss / batches})
                avg_loss = total_loss / batches
                progress_bar.write(f"Validation Loss: {avg_loss}, validation_size={batches*batch_size}, batch_idx={batch_idx}, stochastic_rounding={BitLinear.default_stochastic_rounding}")
                wandb.log({'batch_idx':batch_idx, 'validation_loss': avg_loss, 'validation_size':batches*batch_size, "stochastic_rounding": BitLinear.default_stochastic_rounding})
                key = "round"
                if BitLinear.default_stochastic_rounding:
                    key = "stochastic"
                wandb.log({'batch_idx':batch_idx, f'validation_loss_'+key: avg_loss, 'validation_size':batches*batch_size, "stochastic_rounding": BitLinear.default_stochastic_rounding})

        if return_to_train:
            model.train()

    def sample_output2(model, batch_idx=-1, validation_size=0):
        wandb.log({'weights':get_weight_distribution(model), 'batch_idx':batch_idx, "stochastic_rounding": BitLinear.default_stochastic_rounding})
        if QF == QF_3: # rounding does nothing when weights are already quantized
            sample_output(model, batch_idx, validation_size)
            return
        old_rounding = BitLinear.default_stochastic_rounding
        BitLinear.default_stochastic_rounding = False
        sample_output(model, batch_idx, validation_size)
        BitLinear.default_stochastic_rounding = True
        sample_output(model, batch_idx, validation_size)
        BitLinear.default_stochastic_rounding = old_rounding

    # Train the model
    model.train()
    lowest_loss = float('inf')  # Initialize lowest loss as infinity
    recent_losses = deque([0]*10000, maxlen=10000)
    recent_loss_100 = 0
    recent_loss_1000 = 0
    recent_loss_10000 = 0

    train_loader = get_training_loader(train_subset,max_length)

    for epoch in range(1):  # Number of epochs
        model.save_pretrained(f'{model_save_path}/e{epoch}')
        total_loss = 0.0  # Initialize total loss for this epoch
        progress_bar = tqdm(train_loader, desc=f"E {epoch + 1}", position=0, leave=True)
        n = 1
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            #print(batch)
            #print(batch.keys())
            #print(type(batch['input_ids']))
            #max_id = batch['input_ids'].max().item()
            #if max_id >= model.get_input_embeddings().num_embeddings:
            #    print(f"Max input id ({max_id}) is out of range ({model.get_input_embeddings().num_embeddings})")
            loss = calculate_loss(model, batch['input_ids'].to(device))
            loss.backward()
            optimizer.step()
            
            quantize_weights(model, qf=QF)
            
            current_loss = loss.item()
            total_loss += current_loss
            last_removed = recent_losses[0]
            recent_losses.append(current_loss)
            recent_loss_10000 += current_loss - last_removed
            recent_loss_1000 += current_loss - recent_losses[-1001]
            recent_loss_100 += current_loss - recent_losses[-101]

            #print(total_loss, batch_idx, batch_size)
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'a': avg_loss, 'c': current_loss,
                'a2': recent_loss_100/100, 'a3': recent_loss_1000/1000, 'a4': recent_loss_10000/10000})
            tokens = (batch_idx+1)*batch_size*max_length
            compute_cost = tokens * cost
            
            scheduler.record_lose(current_loss)
            
            wandb.log({'tokens':tokens,'batch_idx':batch_idx, 'compute_cost':compute_cost,
                       'loss': current_loss, 'avg_loss': avg_loss, 
                       'recent_loss_100': recent_loss_100/100, 'recent_loss_1000': recent_loss_1000/1000, 'recent_loss_10000': recent_loss_10000/10000,
                       'scheduler':scheduler.get_states(),
                       "stochastic_rounding": BitLinear.default_stochastic_rounding})
            
            scheduler.step()
            
            if batch_idx % ((512*(2**n)//batch_size)) == 0:
                n += 1
                sample_output2(model, batch_idx, min(batch_idx*batch_size//8, validation_size))
            
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss}")
    # Save the model
    model.save_pretrained(f'{model_save_path}/finish')

    print('Finished Training')
    sample_output2(model,-1,validation_size)
    wandb.finish()

# only run if main
if __name__ == "__main__":
    #train(tiny_stories_ref(hidden_size=512),"tiny_stories_hs_512")
    #train(tiny_stories_ref(hidden_size=1024),"tiny_stories_hs_1024")
    #train(tiny_stories_ref(hidden_size=2048),"tiny_stories_hs_2048")
    #train(bitnet_ref(), "bitnet_ref")
    for rounds in [16,64,256]:
        for hidden_size in [16,32,64,128,256,512]:
            train_subset = rounds*1024*4
            #train(tiny_stories_ref(hidden_size=hidden_size),"tiny_stories_hs_"+str(hidden_size),hidden_size*2, train_subset=train_subset)
            train(llama_ref(hidden_size=hidden_size),"llama_hs_"+str(hidden_size),hidden_size*2, train_subset=train_subset)
            train(bitnet_ref(hidden_size=hidden_size),"bitnet_hs_"+str(hidden_size),hidden_size*2, train_subset=train_subset)
