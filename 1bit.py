# taken from hf-mirror.com/1bitLLM
from modeling_bitnet import BitnetForCausalLM 
from tokenization_bitnet import BitnetTokenizer 

from typing import List, Dict
import re
import time
import torch
torch.manual_seed(0)

localFilesOnly = False # set to False to retrieve from Hugging Face model hub
path = '1bitLLM/bitnet_b1_58-large' # pretrained 729M params trinary model

device = torch.device("cpu")
model = None
if torch.cuda.is_available():
    device = torch.device("cuda")
    model = BitnetForCausalLM.from_pretrained(
        path,
        device_map=device,
        torch_dtype=torch.float16,
        local_files_only=localFilesOnly,
    ).half()
else:
    model = BitnetForCausalLM.from_pretrained(
        path,
        device_map=device,
        #torch_dtype=torch.float16, # CPU does not support float16
        local_files_only=localFilesOnly,
    )#.half()  
    
tokenizer = BitnetTokenizer.from_pretrained(pretrained_model_name_or_path=path, local_files_only=localFilesOnly)



# modified from modeling_minicpm.py
@torch.inference_mode()
def chat(model, tokenizer, query: str, history: List[Dict] = None, role: str = "user",
             max_length: int = 4096, num_beams=1, do_sample=True, top_p=0.8, temperature=0.3, logits_processor=None,
             **kwargs):
    if history is None:
        history = []
    if do_sample:
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                    "temperature": temperature, "logits_processor": logits_processor, **kwargs}
    else:
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample,
                      "logits_processor": logits_processor, **kwargs}
    #gen_kwargs["repetition_penalty"] = 1.2
    history.append({"role": role, "content": query})
    history_str = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=False)
    #inputs = tokenizer(history_str, return_tensors='pt').to(self.device)
    inputs = tokenizer(history_str, return_tensors='pt').to(device)
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
    response = tokenizer.decode(outputs)
    pattern = re.compile(r".*?(?=<AI>|<用户>)", re.DOTALL) 
    matches = pattern.findall(response)
    if len(matches) > 0:
        response = matches[0]
    history.append({"role": "assistant", "content": response})
    return response, history

for question in ["calculate 1+1", "what is 9*9", "solve 12/9"]:
    t1 = time.time()
    response, history = chat(model, tokenizer, question, max_length=64)
    t2 = time.time()
    print(f'{t2 - t1}s ', question,"\n", response)

