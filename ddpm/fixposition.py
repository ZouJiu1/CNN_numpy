import numpy as np
import os
import sys
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
sys.path.append(abspath)
from net.Embedding import Embedding_layer
import matplotlib.pyplot as plt

class fix_postion_layer():
    def __get_param__(self):
        embedding = np.zeros((self.n_steps, self.embed_dim))
        row = np.array([1/ 10000 ** (2 * j / self.embed_dim) for j in range(self.embed_dim)])
        row = np.reshape(row, (1, self.embed_dim))
        col = np.arange(self.n_steps).reshape((self.n_steps, 1))
        embedding[:, ::2] = np.sin(col * row[:, ::2])
        embedding[:, 1::2] = np.cos(col * row[:, ::2])
        # plt.imshow(embedding)
        # plt.savefig(r'C:\Users\10696\Desktop\access\numpy_cnn\ddpm\fixposi.png', bbox_inches='tight')
        # plt.show()
        # plt.close()
        return embedding
        
    def __init__(self, n_steps, embed_dim):
        self.n_steps = n_steps
        self.embed_dim = embed_dim
        self.param = self.__get_param__()
        self.fix_layer = Embedding_layer(n_steps, embed_dim, self.param)

    def forward(self, inputs):
        out = self.fix_layer.forward(inputs)
        return out

    def backward(self, delta):
        # delta = self.fix_layer.backward(delta)
        return delta

    def update(self, lr):
        # self.fix_layer.update(lr)
        pass

    def setzero(self):
        # self.fix_layer.setzero()
        pass

    def save_model(self):
        return [self.fix_layer.save_model()]

    def restore_model(self, models):
        self.fix_layer.restore_model(models[0])

if __name__=="__main__":
    n_steps = 1000
    embed_dim  = 100
    posit = fix_postion_layer(n_steps, embed_dim)