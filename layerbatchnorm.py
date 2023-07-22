# https://zhuanlan.zhihu.com/p/642043155
import numpy as np
import torch 
from torch import nn
from copy import deepcopy

def torch_compare_batchnorm(num_feature, inputs, gamma, beta, affine, delta=''):
    network = nn.BatchNorm2d(num_feature, affine = affine).requires_grad_(True)
    network.double()
    cnt = 0
    for i in network.parameters():
        if cnt==0:
            i.data = torch.from_numpy(gamma)
            i.retain_grad = True
        else:
            i.data = torch.from_numpy(beta)
            i.retain_grad = True
        cnt += 1
    inputs = torch.tensor(inputs, requires_grad=True, dtype=torch.float64)
    output = network(inputs)
    delta = torch.tensor(delta)
    output.backward(delta)
    # sum = torch.sum(output) # make sure the gradient is 1
    # kk = sum.backward()
    grad_gamma = 0
    grad_beta   = 0
    cnt = 0
    for i in network.parameters():
        if cnt==0:
            grad_gamma = i.grad
        else:
            grad_beta = i.grad
        cnt += 1
    inputs.retain_grad()
    output.retain_grad()
    k = inputs.grad
    return output, k, grad_gamma, grad_beta

class layer_batchnorm(object):
    def __init__(self, num_feature, train = True, affine = True, gamma = [], beta = []):
        self.num_feature = num_feature
        self.affine = affine
        self.gamma = np.ones(num_feature)
        self.beta  = np.zeros(num_feature)
        if affine and list(gamma)!=[]:
            self.gamma = gamma

        if affine and list(beta)!=[]:
            self.beta = beta

        self.gamma_delta = np.zeros(num_feature).astype(np.float64)
        self.beta_delta = np.zeros(num_feature).astype(np.float64)
        self.running_mean = np.zeros(num_feature).astype(np.float64)
        self.running_var = np.zeros(num_feature).astype(np.float64)
        self.state = 0
        self.ep = 1e-5
        self.train = train

    def forward(self, inputs):
        self.inputs = deepcopy(inputs)
        self.inshape = inputs.shape
        self.mean = np.mean(inputs, axis = (0, 2, 3))
        self.var  = np.var(inputs, axis=(0, 2, 3))
        if self.state==0:
            self.running_mean = self.mean
            self.running_var  = self.var
            self.state = 6
        else:
            self.running_mean = 0.1 * self.mean + 0.9 * self.running_mean
            self.running_var = 0.1 * self.var + 0.9 * self.running_var
        if self.train:
            outputs = (inputs - self.mean[np.newaxis, :, np.newaxis, np.newaxis]) / \
                np.sqrt(self.var[np.newaxis, :, np.newaxis, np.newaxis] + self.ep)
        else:
            outputs = (inputs - self.running_mean[np.newaxis, :, np.newaxis, np.newaxis]) / \
                np.sqrt(self.running_var[np.newaxis, :, np.newaxis, np.newaxis] + self.ep)
        self.normal = deepcopy(outputs)
        if self.affine:
            outputs = outputs * self.gamma[np.newaxis, :, np.newaxis, np.newaxis] + \
                self.beta[np.newaxis, :, np.newaxis, np.newaxis]
        return outputs

    def backward(self, delta, lr = 1e-10):
        # previous layer delta
        N, C, H, W = self.inshape
        NHW = N * H * W

        if not self.affine:
            self.gamma = np.ones(self.num_feature)
        mean = self.mean[np.newaxis, :, np.newaxis, np.newaxis]
        gamma = self.gamma[np.newaxis, :, np.newaxis, np.newaxis]
        var = (self.var + self.ep)[np.newaxis, :, np.newaxis, np.newaxis]

        partone   = (gamma / np.sqrt(var) / NHW)
        parttwo   = delta * NHW
        partthree = np.sum(delta, axis=(0, 2, 3), keepdims=True)
        parttail  =  np.sum(delta * (self.inputs - mean), axis=(0, 2, 3), keepdims=True)
        parttail  = (1/var) * (self.inputs - mean) * parttail
        input_delta = partone * (parttwo - partthree - parttail)

        # mean = self.mean[np.newaxis, :, np.newaxis, np.newaxis]
        # gamma = self.gamma[np.newaxis, :, np.newaxis, np.newaxis]
        # var = (self.var + self.ep)[np.newaxis, :, np.newaxis, np.newaxis]
        # one = delta * gamma
        # two = np.sum(-0.5 * one * (self.inputs - mean), axis=(0,2,3), keepdims=True) * (var**(-1.5))
        # three = np.sum((-1.0/np.sqrt(var)) * one, axis=(0,2,3), keepdims=True) + np.sum(two*(-2*(self.inputs - mean)), axis=(0,2,3), keepdims=True)/NHW
        # input_delta = one/np.sqrt(var) + \
        #     (2 * two * (self.inputs - mean)/NHW) + \
        #     three/NHW
            
        if self.affine:
            self.gamma_delta = np.sum(self.normal * delta, axis = (0, 2, 3))
            self.beta_delta = np.sum(delta, axis = (0, 2, 3))
            self.gamma -= self.gamma_delta * lr
            self.beta  -= self.beta_delta * lr

        return input_delta

    def save_model(self):
        return [self.gamma, self.beta, self.running_mean, self.running_var]

    def restore_model(self, models):
        self.gamma = models[0]
        self.beta = models[1]
        self.running_mean = models[2]
        self.running_var = models[3]

def train_single():
    inputs = np.random.rand(2, 3, 60, 60).astype(np.float64)
    outputs = np.random.rand(2, 3, 60, 60).astype(np.float64)
    affine = True
    num_feature = inputs.shape[1]
    gamma = np.random.rand(num_feature).astype(np.float64)
    beta = np.random.rand(num_feature).astype(np.float64)

    batchnorm = layer_batchnorm(num_feature=num_feature, affine=affine, gamma=gamma, beta=beta)
    for i in range(3000):
        out = batchnorm.forward(inputs)
        sum = np.sum((inputs - out) * (inputs - out))
        delta = 2*(out - inputs)
        partial = batchnorm.backward(delta, 0.0000001)
        print(sum)

if __name__=="__main__":
    # train_single()
    inputs = np.random.rand(100, 100, 30, 30).astype(np.float64)
    affine = True
    num_feature = inputs.shape[1]
    gamma = np.ones(num_feature) #np.random.rand(num_feature).astype(np.float64)
    beta = np.zeros(num_feature) #np.random.rand(num_feature).astype(np.float64)

    batchnorm = layer_batchnorm(num_feature=num_feature, affine=affine, gamma=gamma, beta=beta)
    output = batchnorm.forward(inputs)
    # delta = np.ones(inputs.shape).astype(np.float64)
    delta = np.random.rand(inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]).astype(np.float64)
    partial = batchnorm.backward(delta)
    
    output_torch, partial_torch, grad_gamma_torch, grad_beta_torch = torch_compare_batchnorm(num_feature, inputs, gamma, beta, affine, delta)
    assert np.mean(np.abs(output - output_torch.cpu().detach().numpy())) < 1e-3, np.mean(np.abs(output - output_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(partial - partial_torch.cpu().detach().numpy())) < 1e-3, np.mean(np.abs(partial - partial_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(batchnorm.gamma_delta - grad_gamma_torch.cpu().detach().numpy())) < 1e-3, np.mean(np.abs(batchnorm.gamma_delta - grad_gamma_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(batchnorm.beta_delta - grad_beta_torch.cpu().detach().numpy())) < 1e-3, np.mean(np.abs(batchnorm.beta_delta - grad_beta_torch.cpu().detach().numpy()))