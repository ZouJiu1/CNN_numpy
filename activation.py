import numpy as np
from copy import deepcopy

class ReLU(object):

    def forward(self, inputs):
        self.inputs = deepcopy(inputs)
        inputs[inputs < 0 ] = 0
        return inputs    

    def backward(self, delta, lr = ''):
        return (self.inputs > 0) * delta

if __name__=="__main__":
    ReLU()