import math
import torch
from torch import nn


def weight_quant(weight, num_bits=1):
    dtype = weight.dtype
    weight = weight.float()
    s =  1 / weight.abs().mean().clamp(min=1e-5)
    result = (weight * s).round().clamp(-1, 1) / s
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

class BitLinearOLD(nn.Linear): # original code 

    def __init__(self,
            *kargs,
            weight_bits=1,
            input_bits=8,
            **kwargs
        ):
        super(BitLinear, self).__init__(*kargs, **kwargs)
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

class BitLinear(nn.Linear):

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

    def forward(self, input):
        if self.training:
            quant_input = STEQuantize_activation_quant_8.apply(input)
            quant_weight = STEQuantize_weight_quant.apply(self.weight)
            self.quant_weight = None # quant_weight can not be reused during training
        else:
            quant_input = activation_quant(input)
            # quantize weight only once for inference
            if self.quant_weight is None:
                self.quant_weight = weight_quant(self.weight)
            quant_weight = self.quant_weight

        out = nn.functional.linear(quant_input, quant_weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out