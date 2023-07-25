import numpy as np

class flatten_layer():
    def forward(self, inputs):
        self.shape = inputs.shape
        return np.stack([i.flatten() for i in inputs])
    
    def backward(self, delta):
        return np.reshape(delta, self.shape)

    def setzero(self):
        pass

    def update(self, lr = 1e-10):
        pass

    def __name__(self):
        return "flatten_layer"