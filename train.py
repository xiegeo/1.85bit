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
from utils_quant import BitLinear, quantize_weights

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



def get_data_loader(dataset_type,train_subset, max_length, shuffle=True):
    sfn = f"tokenized_{dataset_type}_dataset_{max_length}"
    if tokenizer_path == "1bitLLM/bitnet_b1_58-large":
        sfn = f"bitnet_{dataset_type}_tokens_{max_length}"
    if not os.path.exists(sfn) and os.path.exists("../"+sfn): # find the file if it's in the parent directory
        sfn = "../"+sfn
    if os.path.exists(sfn):
        tokenized_dataset_full = load_from_disk(sfn)
    else:
        # use the TinyStories dataset
        dataset = load_dataset('roneneldan/TinyStories')
        print(dataset.keys())
        if dataset_type not in dataset:
            raise ValueError(f"dataset type {dataset_type} not found in dataset {dataset.keys()}")
        sub_dataset = dataset[dataset_type]
        tokenized_dataset_full = sub_dataset.map(
            lambda x: tokenizer(
                x['text'], padding="max_length", max_length=max_length, truncation=True, return_tensors='pt'
            ), batched=True)
        tokenized_dataset_full.save_to_disk(sfn)
    tokenized_dataset_full.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    print(f"use {dataset_type} dataset with max_length={max_length}, train_subset={train_subset}, number of tokens={max_length*train_subset}")
    tokenized_dataset = torch.utils.data.Subset(tokenized_dataset_full, indices=range(train_subset))

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
    def __init__(self, optimizer: Optimizer, lr_decay=0.3, slow_start=1, swap_width=100, swaps=5, lr_min=1e-6):
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.lr_max = self.base_lrs[0] 
        self.current_lr = self.lr_max * slow_start
        self.lr_decay = lr_decay
        self.swap_width = swap_width
        self.swaps = swaps
        self.decision_steps = 2 * swap_width * swaps
        self.lr_min = lr_min
        self.last_loss = 0
        self.loss_avg = [0,0]
        self.loss_count = 0
        self.loss_weight_scale = 2/((self.swap_width+1)*self.swap_width*self.swaps) # 1/(1+2+3+...+swap_width)*swaps
        super(DynamicLearningRate, self).__init__(optimizer)
    
    def higher_lose(self):
        return min(self.current_lr/self.lr_decay, self.lr_max)
    
    def lower_lose(self):
        return max(self.current_lr*self.lr_decay, self.lr_min)

    def lose_index(self):
        return (self.loss_count//self.swap_width +1)%2 # lower goes first, test for plateau
    
    def lose_weight(self):
        return (self.loss_count%self.swap_width+1)*self.loss_weight_scale
    
    def record_lose(self, loss):
        self.loss_avg[self.lose_index()] += loss * self.lose_weight()
        self.loss_count += 1
        if self.loss_count == self.decision_steps:
            if self.loss_avg[1]<self.loss_avg[0]:
                self.update_lr(self.lower_lose())
            else:
                self.update_lr(self.higher_lose())
            
    def update_lr(self, new_lr):
        if new_lr != self.current_lr:
            tqdm.write(f'update lr to {new_lr}')
        self.current_lr = new_lr
        self.loss_avg = [0,0]
        self.loss_count = 0
    
    def get_lr(self):
        if self.lose_index() == 0: 
            return [self.higher_lose()]
        return [self.lower_lose()]
    
    def get_states(self):
        return {
            'current_lr':self.current_lr,
            'loss_avg_0':self.loss_avg[0],
            'loss_avg_1':self.loss_avg[1],
        }

def train(model,model_name, cost, train_subset = 1024*16, max_length=64, optimizer_function=AdamWFun(), QW=False):
    
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
        validation_size = (100//batch_size) * batch_size

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
                "quantize_training_weights": QW
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
        if return_to_train:
            model.train()

    def sample_output2(model, batch_idx=-1, validation_size=0):
        if QW: # rounding does nothing when weights are already quantized
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
            if QW:
                quantize_weights(model)
            
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
                       'scheduler':scheduler.get_states()})
            
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
