import numpy as np

class myddpm_noise(object):
    def __init__(self, n_steps, embed_dim, minbeta = 0.0001, maxbeta = 0.03, shape = (1, 30 - 2, 30 - 2)):
        self.n_steps = n_steps
        self.embed_dim = embed_dim
        self.beta = np.linspace(minbeta, maxbeta, n_steps)
        self.alpha = 1 - self.beta
        self.prod_alpha = np.array([np.prod(self.alpha[: i + 1]) for i in range(len(self.alpha))])
    
    def forward(self, inputs, fixposition = None, etamul = None):
        n, c, h, w = inputs.shape

        aprodd = self.prod_alpha[fixposition]
        
        noisy = np.sqrt(aprodd).reshape(n, 1, 1, 1) * inputs + np.sqrt((1 - aprodd)).reshape(n, 1, 1, 1) * etamul

        return noisy

if __name__ == "__main__":
    myn = myddpm_noise(1000, 100)
    inputs = np.random.rand(2, 2, 20 ,20)
    etamul = np.random.rand(2, 2, 20 ,20)
    fixposition = np.random.randint(0, 1000, 2)
    myn.forward(inputs, fixposition, etamul)