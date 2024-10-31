#!python 1.85bit/1bit.py
#!python 1bit.py
#!python 1.85bit/train.py
#!python train.py

from models import tiny_stories_ref, bitnet_ref, llama_ref
from utils_quant import BitLinear
from train import train
#train(tiny_stories_ref(layers=2, hidden_size=256),"tiny_stories_hs_256")
#train(tiny_stories_ref(layers=2, hidden_size=512),"tiny_stories_hs_512")
#train(tiny_stories_ref(layers=4, hidden_size=128),"tiny_stories_l_4_hs_128")
for rounds in []:
        for hidden_size in [16,32,64,128,256,512]:
            #if rounds > hidden_size//16:
            #    continue
            train_subset = rounds*1024*1024//256 #rounds*1024*512//hidden_size
            #train(tiny_stories_ref(hidden_size=hidden_size),"tiny_stories_hs_"+str(hidden_size),hidden_size*2, train_subset=train_subset)
            train(llama_ref(hidden_size=hidden_size),"llama_hs_"+str(hidden_size),hidden_size*2, train_subset=train_subset)
            train(bitnet_ref(hidden_size=hidden_size),"bitnet_hs_"+str(hidden_size),hidden_size*2, train_subset=train_subset)


for rounds in [1,8]:
    for layers in [1,2,3,4]:
        for hidden_size in [128,256,512]:
            #if rounds > hidden_size//16:
            #    continue
            train_subset = rounds*1024*1024//64 #rounds*1024*512//hidden_size
            name = f'_l_{layers}_hs_{hidden_size}'
            train(llama_ref(hidden_size=hidden_size, layers=layers),"llama"+name,hidden_size*layers, train_subset=train_subset)
            train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet"+name,hidden_size*layers, train_subset=train_subset)
            BitLinear.default_stochastic_rounding = True
            train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet_s"+name,hidden_size*layers, train_subset=train_subset)
            BitLinear.default_stochastic_rounding = False
            
for rounds in [1,32]:
    train_subset = rounds*1024*1024//64 
    layers = 1
    hidden_size = 512
    name = f'_l_{layers}_hs_{hidden_size}'
    train(llama_ref(hidden_size=hidden_size, layers=layers),"llama"+name,hidden_size*layers, train_subset=train_subset)
    train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet"+name,hidden_size*layers, train_subset=train_subset)

    layers = 2
    hidden_size = 256
    name = f'_l_{layers}_hs_{hidden_size}'
    train(llama_ref(hidden_size=hidden_size, layers=layers),"llama"+name,hidden_size*layers, train_subset=train_subset)
    train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet"+name,hidden_size*layers, train_subset=train_subset)

    layers = 2
    hidden_size = 512
    name = f'_l_{layers}_hs_{hidden_size}'
    train(bitnet_ref(hidden_size=hidden_size, layers=layers),"bitnet"+name,hidden_size*layers, train_subset=train_subset)

