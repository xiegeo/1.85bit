import math
import torch
from torch import nn

printed_layers = set()        

def QF_noop(weight):
    return weight
    
def QF_3(weight):
    return stochastic_weight_quant_no_scale(weight)

def QF_8b(weight): # quantize the weight to 8 bits using the same algorithm as activation_quant_8, but with stochastic rounding
    dtype = weight.dtype
    x = weight.float()
    Qn = -2 ** (8 - 1)
    Qp = 2 ** (8 - 1) - 1
    s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * + torch.rand_like(weight)).floor().clamp(Qn, Qp) / s
    return result.type(dtype)   

def QF_4b(weight): 
    dtype = weight.dtype
    x = weight.float()
    Qn = -2 ** (4 - 1)
    Qp = 2 ** (4 - 1) - 1
    s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * + torch.rand_like(weight)).floor().clamp(Qn, Qp) / s
    return result.type(dtype)

def QF_3b(weight):
    dtype = weight.dtype
    x = weight.float()
    Qn = -2 ** (2 - 1)
    Qp = 2 ** (2 - 1) - 1
    s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * + torch.rand_like(weight)).floor().clamp(Qn, Qp) / s
    return result.type(dtype)

def QF_2b(weight):
    dtype = weight.dtype
    x = weight.float()
    Qn = -2 ** (2 - 1)
    Qp = 2 ** (2 - 1) - 1
    s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * + torch.rand_like(weight)).floor().clamp(Qn, Qp) / s
    return result.type(dtype)

def quantize_weights(model: nn.Module, qf):
    if qf == QF_noop:
        return
    global printed_layers
    for layer in model.modules():
        if type(layer) in [BitLinear]:
            layer.weight.data = qf(layer.weight.data)
            if type(layer) not in printed_layers:
                print(f"[Y] Layer {type(layer)} weights are quantized")
                printed_layers.add(type(layer))

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