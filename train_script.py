#!python 1.85bit/1bit.py
#!python 1bit.py
#!python 1.85bit/train.py
#!python train.py

from models import tiny_stories_ref, bitnet_ref, llama_ref
from utils_quant import BitLinear
from train import train


for rounds in [1,4,16]:
    for layers in [1,4]:
        hs = [128,512]
        if layers == 4:
            hs = [128]
        for hidden_size in hs:
            #if rounds > hidden_size//16:
            #    continue
            train_subset = rounds*1024*1024//64 #rounds*1024*512//hidden_size
            name = f'_l_{layers}_hs_{hidden_size}'
            #BitLinear.default_stochastic_rounding = True
            train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet_qwl"+name,hidden_size*layers, train_subset=train_subset, training_mode='qw',lr=1e-3)
            train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet_qwl.2"+name,hidden_size*layers, train_subset=train_subset, training_mode='qw',lr=2e-4)
            #train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet_s"+name,hidden_size*layers, train_subset=train_subset,)
            #train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet_sgd.5_s"+name,hidden_size*layers, train_subset=train_subset, training_mode='sgd', lr=5e-4)
            BitLinear.default_stochastic_rounding = False
            # train(llama_ref(hidden_size=hidden_size, layers=layers),"llama"+name,hidden_size*layers, train_subset=train_subset)
            # train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet"+name,hidden_size*layers, train_subset=train_subset)
            