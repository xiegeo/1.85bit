import math
import torch
from torch import nn
import numpy as np
from scipy.stats import entropy
import wandb

printed_layers = set()        

def QF_noop(weight):
    return weight
    
def QF_3(weight):
    return stochastic_weight_quant_no_scale(weight)

def QF_3_top(weight):
    dtype = weight.dtype
    weight = weight.float().clamp(-1, 1)
    rw = weight.round()
    d = weight - rw
    ad = d.abs()
    
    # Get the number of weights to change to much the learning rate
    ad_sum = ad.view(-1).sum()
    k = (ad_sum + torch.rand(1, device=weight.device)).floor().long().item()
    
    # Get the top k indices
    if k > 0:
        selected = ad.view(-1).topk(k, largest=True).indices
        
        # Update rw based on selected indices
        rw_flat = rw.view(-1)
        d_flat = d.view(-1)
        rw_flat[selected] += d_flat[selected].sign()
    
    return rw.type(dtype)

def QF_8b(weight): 
    dtype = weight.dtype
    x = weight.float()
    Qn = -2 ** (8 - 1)
    Qp = 2 ** (8 - 1) - 1
    s = Qp #/ x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * s + torch.rand_like(weight)).floor().clamp(Qn, Qp) / s
    return result.type(dtype)   

def QF_4b(weight): 
    dtype = weight.dtype
    x = weight.float()
    Qn = -2 ** (4 - 1)
    Qp = 2 ** (4 - 1) - 1
    s = Qp #/ x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * s + torch.rand_like(weight)).floor().clamp(Qn, Qp) / s
    return result.type(dtype)

def QF_3b(weight):
    dtype = weight.dtype
    x = weight.float()
    Qn = -2 ** (2 - 1)
    Qp = 2 ** (2 - 1) - 1
    s = Qp #/ x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * s + torch.rand_like(weight)).floor().clamp(Qn, Qp) / s
    return result.type(dtype)

def QF_2b(weight):
    dtype = weight.dtype
    x = weight.float()
    Qn = -2 ** (2 - 1)
    Qp = 2 ** (2 - 1) - 1
    s = Qp #/ x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * s + torch.rand_like(weight)).floor().clamp(Qn, Qp) / s
    return result.type(dtype)

def quantize_weights(model: nn.Module, qf):
    if qf == QF_noop:
        return
    global printed_layers
    for layer in model.modules():
        if type(layer) in [BitLinear, nn.Linear]:
            layer.weight.data = qf(layer.weight.data)
            if type(layer) not in printed_layers:
                print(f"[Y] Layer {type(layer)} weights are quantized")
                printed_layers.add(type(layer))
                
def get_weight_distribution(model: nn.Module):
    collection = {}
    all_weights = []
    zeros = 0
    round_zeros = 0
    for name, layer in model.named_modules():
        if type(layer) in [BitLinear,nn.Linear]:
            weights = layer.weight.data.cpu().numpy().flatten()
            all_weights.extend(weights)
            collection[name + "_h"] = wandb.Histogram(weights)
            zero_count = (layer.weight.data == 0).sum().item() 
            # ratio of weights that are zero
            collection[name + "_0r"] = zero_count / layer.weight.numel()
            round_zero_count = (layer.weight.data.round() == 0).sum().item()
            # ratio of weights that round to zero
            collection[name + "_r0r"] = round_zero_count / layer.weight.numel()
            zeros += zero_count
            round_zeros += round_zero_count
    collection["all_h"] = wandb.Histogram(all_weights)
    collection["all_0r"] = zeros / len(all_weights)
    collection["all_r0r"] = round_zeros / len(all_weights)
    
    # Calculate the entropy for discrete weights up to 7 decimal bits 
    value_counts = np.bincount(np.array(np.round(all_weights*(2**7)), dtype=int)+1024)
    collection["all_entropy_c8"] = entropy(value_counts, base=2)
    # Calculate the information density (entropy) for floating-point weights
    hist, bin_edges = np.histogram(all_weights, bins='auto', density=True)
    hist += 1e-10  # Add a small constant to avoid log(0)
    collection["all_entropy_f"] = entropy(hist, base=2)
    
    collection["all_abs_avg"] = np.mean(np.abs(all_weights))
    collection["all_mean"] = np.mean(all_weights)
    collection["all_std"] = np.std(all_weights)
    
    return collection


def weight_quant(weight, num_bits=1):
    dtype = weight.dtype
    weight = weight.float()
    s =  1 / weight.abs().mean().clamp(min=1e-5)
    result = (weight * s).round().clamp(-1, 1) / s
    return result.type(dtype)

def stochastic_weight_quant(weight):
    dtype = weight.dtype
    weight = weight.float()
    s =  1 / weight.abs().mean().clamp(min=1e-5)
    result = (weight * s + torch.rand_like(weight)).floor().clamp(-1, 1) / s
    return result.type(dtype)

def stochastic_weight_quant_no_scale(weight):
    dtype = weight.dtype
    weight = weight.float()
    result = (weight + torch.rand_like(weight)).floor().clamp(-1, 1)
    return result.type(dtype)


def activation_quant(x, num_bits=8):
    dtype = x.dtype
    x = x.float()
    Qn = -2 ** (num_bits - 1)
    Qp = 2 ** (num_bits - 1) - 1
    s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * s).round().clamp(Qn, Qp) / s
    return result.type(dtype)   

def activation_quant_8(x):
    dtype = x.dtype
    x = x.float()
    Qn = -2 ** (8 - 1)
    Qp = 2 ** (8 - 1) - 1
    s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * s).round().clamp(Qn, Qp) / s
    return result.type(dtype)   



class BitLinearOLD(nn.Linear): # original code, not in use

    def __init__(self,
            *kargs,
            weight_bits=1,
            input_bits=8,
            **kwargs
        ):
        super(BitLinearOLD, self).__init__(*kargs, **kwargs)
        """
        RMSNorm is placed outside BitLinear
        """
        self.weight_bits = weight_bits
        self.input_bits = input_bits

    def forward(self, input):
        
        quant_input = input + (activation_quant(input, self.input_bits) - input).detach()
        quant_weight = self.weight + (weight_quant(self.weight, self.weight_bits) - self.weight).detach()

        out = nn.functional.linear(quant_input, quant_weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out

class STEQuantize_activation_quant_8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return activation_quant_8(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class STEQuantize_weight_quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return weight_quant(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
class STEQuantize_stochastic_weight_quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return stochastic_weight_quant(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class BitLinear(nn.Linear):
    default_stochastic_rounding = False
    QW = False

    def __init__(self,
            *kargs,
            weight_bits=1,
            input_bits=8,
            **kwargs
        ):
        if weight_bits != 1:
            raise ValueError("weight_bits must be 1")
        if input_bits != 8:
            raise ValueError("input_bits must be 8")
        super(BitLinear, self).__init__(*kargs, **kwargs)
        self.quant_weight = None

    def forward(self, input, stochastic_rounding=None):
        if stochastic_rounding is None:
            stochastic_rounding = self.default_stochastic_rounding
        if self.training:
            quant_input = STEQuantize_activation_quant_8.apply(input)
            if BitLinear.QW:
                quant_weight = self.weight
            elif stochastic_rounding:
                quant_weight = STEQuantize_stochastic_weight_quant.apply(self.weight)
            else:
                quant_weight = STEQuantize_weight_quant.apply(self.weight)
            self.quant_weight = None # quant_weight can not be reused after training
        else:            
            quant_input = activation_quant(input)
            if stochastic_rounding:
                quant_weight = stochastic_weight_quant(self.weight)
            elif self.quant_weight is None:
                # quantize weight only once for inference, if it is stable
                self.quant_weight = weight_quant(self.weight)
                quant_weight = self.quant_weight
            else:
                quant_weight = self.quant_weight

        out = nn.functional.linear(quant_input, quant_weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out