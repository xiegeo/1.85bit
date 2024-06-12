import time, os, json
import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from collections import deque

from modeling_bitnet import BitnetForCausalLM 
from tokenization_bitnet import BitnetTokenizer 
from transformers import AutoTokenizer, AutoModelForCausalLM


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
print(f'use {device}')

tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
# load model from local file
#model = AutoModelForCausalLM.from_pretrained("tiny_stories_ref", local_files_only=True)

model.to(device)
# use the model to generate a story
for text in ["Once","Alice and Bob", "In a galaxy far far away", "The lazy dog"]:
    inputs = tokenizer(text, return_tensors='pt').to(device)
    outputs = model.generate(**inputs, max_length=64)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(json.dumps(decoded))
