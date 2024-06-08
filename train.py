import time, os, json
import torch, wandb
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from collections import deque

from modeling_bitnet import BitnetForCausalLM 
from tokenization_bitnet import BitnetTokenizer 
from transformers import AutoTokenizer

from models import bitnet_64_2, tiny_stories_ref

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
print(f'use {device}')

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

class OnDemandDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.tokenizer(item['text'], padding="max_length", max_length=max_length, truncation=True, return_tensors='pt')

max_length = 64
sfn = f"tokenized_train_dataset_{max_length}"
if not os.path.exists(sfn) and os.path.exists("../"+sfn): # find the file if it's in the parent directory
    sfn = "../"+sfn
if os.path.exists(sfn):
    tokenized_train_dataset_full = load_from_disk(sfn)
else:
    # use the TinyStories dataset
    dataset = load_dataset('roneneldan/TinyStories')
    print(dataset.keys())
    train_dataset = dataset['train']
    #test_dataset = dataset['validation']
    tokenizer.pad_token = tokenizer.eos_token
    #tokenized_train_dataset = OnDemandDataset(train_dataset, tokenizer)
    tokenized_train_dataset_full = train_dataset.map(
        lambda x: tokenizer(
            x['text'], padding="max_length", max_length=max_length, truncation=True, return_tensors='pt'
        ), batched=True)
    tokenized_train_dataset_full.save_to_disk(sfn)
tokenized_train_dataset_full.set_format(type='torch', columns=['input_ids', 'attention_mask'])



train_subset = 1024*128
tokenized_train_dataset = torch.utils.data.Subset(tokenized_train_dataset_full, indices=range(train_subset))
print(f"use training dataset with max_length={max_length}, train_subset={train_subset}, number of tokens={max_length*train_subset}")

# Define your dataloaders
batch_size = 64
if device.type == 'cpu':
    batch_size = min(batch_size, 8)
train_loader = torch.utils.data.DataLoader(tokenized_train_dataset, batch_size=batch_size, shuffle=True)
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# Initialize your model
#model = AutoModel.from_config(AutoConfig.from_dict(bitnet_64_2))
model = tiny_stories_ref().to(device)
model_name = f'tiny_stories_ref_{max_length}'
model_save_path = f'{model_name}/{time.time()}'

# Prepare the optimizer
optimizer = torch.optim.AdamW(model.parameters())

loss_fct = torch.nn.CrossEntropyLoss(reduction="sum").to(device)

def calculate_loss_old(model, input):
    output = model(input,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False)[0]
    shift_logits = output[:, :-1, :].contiguous()
    shift_labels = input[:, 1:]
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

def calculate_loss(model, input):
    output = model(input,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False)[0]
    shift_logits = output[:, :-1, :].contiguous()
    shift_labels = input[:, 1:]
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
    return loss

wandb.init(project="npl185", name=model_name, 
           config={
               "tokenizer_max_length": max_length,
               "train_subset": train_subset,
               "train_batch_size": batch_size, 
               "model_save_path": model_save_path,
               "model_name": model_name,
               "device": device.type,
               "optimizer": optimizer.__class__.__name__,
               "host": os.name,
            })

def sample_output(model, batch_idx=-1):
    for text in ["Once","Alice and Bob", "In a galaxy far far away"]:
        inputs = tokenizer(text, return_tensors='pt').to(device)
        outputs = model.generate(**inputs, max_length=max_length)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(json.dumps(decoded))
        wandb.log({'batch_idx':batch_idx, 'text': text, 'story': json.dumps(decoded)})

# Train the model
model.train()
lowest_loss = float('inf')  # Initialize lowest loss as infinity
recent_losses = deque([0]*10000, maxlen=10000)
recent_loss_100 = 0
recent_loss_1000 = 0
recent_loss_10000 = 0

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
        loss = calculate_loss(model, batch['input_ids'].to(device))
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item() / batch_size / max_length
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
        
        wandb.log({'batch_idx':batch_idx, 'loss': current_loss, 'avg_loss': avg_loss, 'recent_loss_100': recent_loss_100/100, 'recent_loss_1000': recent_loss_1000/1000, 'recent_loss_10000': recent_loss_10000/10000})
        
        if batch_idx % 10**n == 0:
            n += 1
            sample_output(model, batch_idx)
        
    print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss}")
# Save the model
model.save_pretrained(f'{model_save_path}/finish')

print('Finished Training')
sample_output(model)
wandb.finish()