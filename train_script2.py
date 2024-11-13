

from models import tiny_stories_ref, bitnet_ref, llama_ref
from utils_quant import BitLinear, quantize_weights, QF_noop, QF_3, QF_2b, QF_3b, QF_4b, QF_8b, QF_3_top

from train import train, AdamWFun, SGDFun, get_data_loader


for rounds in [64]:
    train_subset = int(rounds*1024*1024//64)
    hidden_sizes = [512]
    layers = 1
    lrs = [0.001]
    for hidden_size in hidden_sizes:
        for lr in lrs:
            name = f'_lr{lr}_L{layers}_hs{hidden_size}'
            train(llama_ref(hidden_size=hidden_size, layers=layers),"ll3m.99top"+name,hidden_size*layers, train_subset=train_subset, optimizer_function=AdamWFun(lr=lr,betas=(0.99,0.999)), QF=QF_3_top)