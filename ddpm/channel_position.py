import numpy as np
import os
import sys
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
sys.path.append(abspath)
from net.fullconnect import fclayer
from net.activation import SiLU

class channel_position_layer():
    def __init__(self, in_dim, out_dim):
        self.fc0 = fclayer(in_dim, out_dim)
        self.silu = SiLU()
        self.fc1 = fclayer(out_dim, out_dim)

    def forward(self, inputs):
        self.inputs = inputs
        out = self.fc0.forward(inputs)
        self.out0 = self.silu.forward(out)
        out = self.fc1.forward(self.out0)
        return out

    def backward(self, delta):
        delta = self.fc1.backward(delta, self.out0)
        delta = self.silu.backward(delta)
        delta = self.fc0.backward(delta, self.inputs)
        return delta

    def update(self, lr):
        self.fc1.update(lr)
        self.fc0.update(lr)

    def setzero(self):
        self.fc1.setzero()
        self.fc0.setzero()

    def save_model(self):
        return [self.fc0.save_model(), self.fc1.save_model()]

    def restore_model(self, models):
        self.fc0.restore_model(models[0])
        self.fc1.restore_model(models[1])

if __name__=="__main__":
    inputs = np.random.randn(10, 20)
    posit = channel_position_layer(20, 60)
    output = posit.forward(inputs)

    outputs = np.random.randn(10, 60)
    for i in range(10000):
        out = posit.forward(inputs)
        sum = np.sum((outputs - out) * (outputs - out))
        delta = 2 * (out - outputs)
        partial = posit.backward(delta)
        posit.update(0.0001)
        posit.setzero()
        print(sum)