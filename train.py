import time, os
import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from modeling_bitnet import BitnetForCausalLM 
from tokenization_bitnet import BitnetTokenizer 
from transformers import AutoTokenizer

from models import bitnet_64_2, tiny_stories_ref

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
print(f'use {device}')

# use the TinyStories dataset
dataset = load_dataset('roneneldan/TinyStories')
print(dataset.keys())
train_dataset = dataset['train']
test_dataset = dataset['validation']

class OnDemandDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.tokenizer(item['text'], padding="max_length", max_length=512, truncation=True, return_tensors='pt')

sfn = "tokenized_train_dataset_512"
if os.path.exists(sfn):
    tokenized_train_dataset = load_from_disk(sfn)
else:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
    #tokenized_train_dataset = OnDemandDataset(train_dataset, tokenizer)
    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenizer(
            x['text'], padding="max_length", max_length=512, truncation=True, return_tensors='pt'
        ), batched=True)
    tokenized_train_dataset.save_to_disk(sfn)
tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])


# Define your transforms and datasets
#transform = transforms.Compose([transforms.ToTensor()])
#train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Define your dataloaders
batch_size = 16
train_loader = torch.utils.data.DataLoader(tokenized_train_dataset, batch_size=batch_size, shuffle=True)
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# Initialize your model
#model = AutoModel.from_config(AutoConfig.from_dict(bitnet_64_2))
model = tiny_stories_ref()
model_save_path = f'tiny_stories_ref/{time.time()}'

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

# Train the model
model.train()
lowest_loss = float('inf')  # Initialize lowest loss as infinity
for epoch in range(10):  # Number of epochs
    total_loss = 0.0  # Initialize total loss for this epoch
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)
    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        #print(batch)
        #print(batch.keys())
        #print(type(batch['input_ids']))
        loss = calculate_loss(model, batch['input_ids'])
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item() / batch_size
        total_loss += current_loss
        #print(total_loss, batch_idx, batch_size)
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({'avg loss': avg_loss, 'current': current_loss})
    print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss}")
# Save the model
model.save_pretrained('path_to_save_directory')
print('Finished Training')