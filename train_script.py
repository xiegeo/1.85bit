#!python 1.85bit/1bit.py
#!python 1bit.py
#!python 1.85bit/train.py
#!python train.py

from models import tiny_stories_ref, bitnet_ref, llama_ref
from utils_quant import BitLinear
from train import train, AdamWFun, SGDFun


for rounds in [1]:
    for layers in [1]:
        hidden_size = 128
        lrs = [1, 10, 0.3, 0.1]
        for lr in lrs:
            #if rounds > hidden_size//16:
            #    continue
            train_subset = rounds*1024*1024//64 #rounds*1024*512//hidden_size
            name = f'_lr{lr}_L{layers}_hs_{hidden_size}'
            BitLinear.default_stochastic_rounding = True
            train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet_s_sgl_qw"+name,hidden_size*layers, train_subset=train_subset, optimizer_function=SGDFun(lr=lr), QW=True)
            #train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet_s_sgd"+name,hidden_size*layers, train_subset=train_subset, optimizer_function=SGDFun(lr=lr), QW=False)
            #train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet_s_qw"+name,hidden_size*layers, train_subset=train_subset, optimizer_function=AdamWFun(lr=lr), QW=True)
            #train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet_s"+name,hidden_size*layers, train_subset=train_subset, optimizer_function=AdamWFun(lr=lr), QW=False)
            BitLinear.default_stochastic_rounding = False
            #train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet"+name,hidden_size*layers, train_subset=train_subset, optimizer_function=AdamWFun(lr=lr), QW=False)
            
for rounds in [1,4,16,64]:
    for layers in [1]:
        hs = [128,512]
        lr = 0.1
        for hidden_size in hs:
            #if rounds > hidden_size//16:
            #    continue
            train_subset = rounds*1024*1024//64 #rounds*1024*512//hidden_size
            name = f'_l_{layers}_hs_{hidden_size}'
            BitLinear.default_stochastic_rounding = True
            #train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet_qwl"+name,hidden_size*layers, train_subset=train_subset, training_mode='qw',lr=1e-3)
            train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet_qwl_s"+name,hidden_size*layers, train_subset=train_subset, optimizer_function=SGDFun(lr=lr), QW=True)
            train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet_sgd_s"+name,hidden_size*layers, train_subset=train_subset, optimizer_function=SGDFun(lr=lr), QW=False)
            #train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet_s"+name,hidden_size*layers, train_subset=train_subset,)
            #train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet_sgd.5_s"+name,hidden_size*layers, train_subset=train_subset, training_mode='sgd', lr=5e-4)
            train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet_s"+name,hidden_size*layers, train_subset=train_subset, optimizer_function=AdamWFun(lr=lr), QW=False)
            BitLinear.default_stochastic_rounding = False
            # train(llama_ref(hidden_size=hidden_size, layers=layers),"llama"+name,hidden_size*layers, train_subset=train_subset)
            train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet_qw"+name,hidden_size*layers, train_subset=train_subset,optimizer_function=AdamWFun(lr=lr), QW=True)
            train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet"+name,hidden_size*layers, train_subset=train_subset,optimizer_function=AdamWFun(lr=lr), QW=False)
            