# https://zhuanlan.zhihu.com/p/642043155
import numpy as np
import torch 
from torch import nn

def torch_compare_dropout(p, inputs):
    network = nn.Dropout2d(p=p).requires_grad_(True)
    inputs = torch.tensor(inputs, requires_grad=True)
    output = network(inputs)
    sum = torch.sum(output) # make sure the gradient is 1
    kk = sum.backward()
    inputs.retain_grad()
    k = inputs.grad
    return output, k

class dropout_layer(object):
    def __init__(self, dropout_probability):
        self.dropout_probability = dropout_probability

    def forward(self, inputs):
        self.inputs = inputs
        # https://stackoverflow.com/questions/54109617/implementing-dropout-from-scratch
        # to drop some feature not all pixel
        if len(inputs.shape) > 3:
            randn = np.random.rand(inputs.shape[0], inputs.shape[1])[:, :, np.newaxis, np.newaxis]
        elif len(inputs.shape) == 3:
            randn = np.random.rand(inputs.shape[0], inputs.shape[1])[:, :, np.newaxis]
        elif len(inputs.shape) == 2:
            randn = np.random.rand(inputs.shape[0], inputs.shape[1])
        self.mask = randn > self.dropout_probability
        output = self.inputs * self.mask
        # inverted dropout
        output = output / (1 - self.dropout_probability)
        return output
    
    def backward(self, delta, lr = ''):
        # #previous layer delta
        next_delta = (delta * self.mask) / (1 - self.dropout_probability)
        return next_delta

    def __name__(self):
        return "dropout_layer"

if __name__=="__main__":
    inputs = np.random.rand(1, 10, 3, 3)
    dropout_probability = 0.2
    
    dropout = dropout_layer(dropout_probability)
    output = dropout.forward(inputs)
    delta = np.ones(inputs.shape)
    partial = dropout.backward(delta)

    output_torch, partial_torch = torch_compare_dropout(dropout_probability, inputs)
    k = 0