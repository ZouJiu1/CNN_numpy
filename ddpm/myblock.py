import numpy as np
import os
import sys
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
sys.path.append(abspath)
from net.Convolution import convolution_layer
from net.ConvTranspose2d import ConvTranspose2d_layer
from net.activation import SiLU
from net.layernorm import layer_norm


class myblock_three_layer():
    def __init__(self, param0, param1, param2):
        self.myblock0 = myblock_layer(*param0)
        self.myblock1 = myblock_layer(*param1)
        self.myblock2 = myblock_layer(*param2)

    def forward(self, inputs):
        out = self.myblock0.forward(inputs)
        out = self.myblock1.forward(out)
        out = self.myblock2.forward(out)
        return out

    def backward(self, delta):
        delta = self.myblock2.backward(delta)
        delta = self.myblock1.backward(delta)
        delta = self.myblock0.backward(delta)
        return delta

    def update(self, lr):
        self.myblock2.update(lr)
        self.myblock1.update(lr)
        self.myblock0.update(lr)

    def setzero(self):
        self.myblock2.setzero()
        self.myblock1.setzero()
        self.myblock0.setzero()

    def save_model(self):
        return [self.myblock2.save_model(), self.myblock1.save_model(), self.myblock0.save_model()]

    def restore_model(self, models):
        self.myblock2.restore_model(models[0])
        self.myblock1.restore_model(models[1])
        self.myblock0.restore_model(models[2])
    
class myblock_layer():
    def __init__(self, shape, in_channel, out_channel, normalize = True, kernel_size=3, stride=1, padding=1):
        self.normalize = normalize
        self.ln = layer_norm(shape)
        self.convolu0 = convolution_layer(in_channel, out_channel, kernel_size, stride, padding)
        self.convolu1 = convolution_layer(out_channel, out_channel, kernel_size, stride, padding)
        self.silu0 = SiLU()
        self.silu1 = SiLU()

    def forward(self, inputs):
        if self.normalize:
            out = self.ln.forward(inputs)
        else:
            out = inputs
        out = self.convolu0.forward(out)
        out = self.silu0.forward(out)
        out = self.convolu1.forward(out)
        out = self.silu1.forward(out)
        return out

    def backward(self, delta):
        delta = self.silu1.backward(delta)
        delta = self.convolu1.backward(delta)
        delta = self.silu0.backward(delta)
        delta = self.convolu0.backward(delta)
        if self.normalize:
            delta = self.ln.backward(delta)
        return delta

    def update(self, lr):
        self.convolu1.update(lr)
        self.convolu0.update(lr)
        self.ln.update(lr)

    def setzero(self):
        self.convolu1.setzero()
        self.convolu0.setzero()
        self.ln.setzero()

    def save_model(self):
        return [self.convolu0.save_model(), self.convolu1.save_model(), self.ln.save_model()]

    def restore_model(self, models):
        self.convolu0.restore_model(models[0])
        self.convolu1.restore_model(models[1])
        self.ln.restore_model(models[2])

    
class downsample_layer():
    def __init__(self, param0, param1):
        self.convolu0 = convolution_layer(*param0)
        self.convolu1 = convolution_layer(*param1)
        self.silu0 = SiLU()

    def forward(self, inputs):
        out = self.convolu0.forward(inputs)
        out = self.silu0.forward(out)
        out = self.convolu1.forward(out)
        return out

    def backward(self, delta):
        delta = self.convolu1.backward(delta)
        delta = self.silu0.backward(delta)
        delta = self.convolu0.backward(delta)
        return delta

    def update(self, lr):
        self.convolu1.update(lr)
        self.convolu0.update(lr)

    def setzero(self):
        self.convolu1.setzero()
        self.convolu0.setzero()

    def save_model(self):
        return [self.convolu0.save_model(), self.convolu1.save_model()]

    def restore_model(self, models):
        self.convolu0.restore_model(models[0])
        self.convolu1.restore_model(models[1])

    
class upsample_layer():
    def __init__(self, param0, param1):
        self.ConvTranspose2d0 = ConvTranspose2d_layer(*param0)
        self.ConvTranspose2d1 = ConvTranspose2d_layer(*param1)
        self.silu0 = SiLU()

    def forward(self, inputs):
        out = self.ConvTranspose2d0.forward(inputs)
        out = self.silu0.forward(out)
        out = self.ConvTranspose2d1.forward(out)
        return out

    def backward(self, delta):
        delta = self.ConvTranspose2d1.backward(delta)
        delta = self.silu0.backward(delta)
        delta = self.ConvTranspose2d0.backward(delta)
        return delta

    def update(self, lr):
        self.ConvTranspose2d0.update(lr)
        self.ConvTranspose2d1.update(lr)

    def setzero(self):
        self.ConvTranspose2d1.setzero()
        self.ConvTranspose2d0.setzero()

    def save_model(self):
        return [self.ConvTranspose2d0.save_model(), self.ConvTranspose2d1.save_model()]

    def restore_model(self, models):
        self.ConvTranspose2d0.restore_model(models[0])
        self.ConvTranspose2d1.restore_model(models[1])

if __name__=="__main__":
    # c = 20
    # inputs = np.random.randn(1, c, 3, 3)
    # posit = upsample_layer((c, c, 2*2, 2, 1), (c, c, 2, 1))
    # outputs = np.random.randn(1, c, 6+1, 6+1)
    # for i in range(10000):
    #     out = posit.forward(inputs)
    #     sum = np.sum((outputs - out) * (outputs - out))
    #     delta = 2 * (out - outputs)
    #     partial = posit.backward(delta)
    #     posit.update(0.0001)
    #     posit.setzero()
    #     print(sum)


    # c = 20
    # inputs = np.random.randn(1, c, 20, 20)
    # posit = downsample_layer((c, c, 2, 1), (c, c, 2*2, 2, 1))
    # outputs = np.random.randn(1, c, 10 - 1, 10 - 1)
    # for i in range(10000):
    #     out = posit.forward(inputs)
    #     sum = np.sum((outputs - out) * (outputs - out))
    #     delta = 2 * (out - outputs)
    #     partial = posit.backward(delta)
    #     posit.update(0.0001)
    #     posit.setzero()
    #     print(sum)


    inputs = np.random.randn(1, 2, 10, 10)
    posit = myblock_three_layer(((1, 2, 10, 10), 2, 2), ((2, 10, 10), 2, 2), ((2, 10, 10), 2, 2))
    outputs = np.random.randn(1, 2, 10, 10)
    for i in range(10000):
        out = posit.forward(inputs)
        sum = np.sum((outputs - out) * (outputs - out))
        delta = 2 * (out - outputs)
        partial = posit.backward(delta)
        posit.update(0.0001)
        posit.setzero()
        print(sum)
