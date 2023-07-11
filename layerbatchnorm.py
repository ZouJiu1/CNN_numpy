# https://zhuanlan.zhihu.com/p/642043155
import numpy as np
import torch 
from torch import nn
from copy import deepcopy

def torch_compare_batchnorm(num_feature, inputs, gamma, beta):
    network = nn.BatchNorm2d(num_feature).requires_grad_(True)
    network.float()
    cnt = 0
    for i in network.parameters():
        if cnt==0:
            i.data = torch.from_numpy(gamma)
            i.retain_grad = True
        else:
            i.data = torch.from_numpy(beta)
            i.retain_grad = True
        cnt += 1
    inputk = torch.from_numpy(inputs)
    inputs = torch.tensor(inputk, requires_grad=True, dtype=torch.float32)
    output = network(inputs)
    sum = torch.sum(output) # make sure the gradient is 1
    kk = sum.backward()
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
    k = inputs.grad
    return output, k, grad_gamma, grad_beta

class layer_batchnorm(object):
    def __init__(self, num_feature, affine = True, gamma = [], beta = []):
        self.num_feature = num_feature
        self.affine = affine
        if affine and list(gamma)!=[]:
            self.gamma = gamma
        else:
            ranges = np.sqrt(6 / (num_feature))
            self.gamma = np.random.uniform(-ranges, ranges, (num_feature))

        if affine and list(beta)!=[]:
            self.beta = beta
        else:
            ranges = np.sqrt(6 / (num_feature))
            self.beta = np.random.uniform(-ranges, ranges, (num_feature))

        self.gamma_delta = np.zeros(num_feature).astype(np.float32)
        self.beta_delta = np.zeros(num_feature).astype(np.float32)
        self.running_mean = np.zeros(num_feature).astype(np.float32)
        self.running_var = np.zeros(num_feature).astype(np.float32)
        self.state = 0
        self.ep = 1e-5

    def forward(self, inputs):
        self.inputs = inputs.copy()
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
        outputs = (inputs - self.mean[np.newaxis, :, np.newaxis, np.newaxis]) / \
            np.sqrt(self.var[np.newaxis, :, np.newaxis, np.newaxis] + self.ep)
        self.normal = outputs.copy()
        if self.affine:
            outputs = outputs * self.gamma[np.newaxis, :, np.newaxis, np.newaxis] + \
                self.beta[np.newaxis, :, np.newaxis, np.newaxis]
        return outputs
    
    def backward(self, delta, lr = 1e-10):
        # previous layer delta
        N, C, H, W = self.inshape
        NHW = N * H * W
        partone = (1 - 1/NHW) * (1 / np.sqrt(self.var[np.newaxis, :, np.newaxis, np.newaxis] + self.ep))
        parttwo0 = self.inputs - self.mean[np.newaxis, :, np.newaxis, np.newaxis]
        parttwo = (1/NHW - 1) * (1/NHW) * (parttwo0 * parttwo0) * (np.power(self.var+self.ep, -3/2)[np.newaxis, :, np.newaxis, np.newaxis])
        # 1 - (1/NHW) * (parttwo * parttwo) * \
                    # (1/(self.var[np.newaxis, :, np.newaxis, np.newaxis] + self.ep))
        # input_delta = partone * parttwo * self.gamma[np.newaxis, :, np.newaxis, np.newaxis] * delta
        input_delta = (partone + parttwo) * self.gamma[np.newaxis, :, np.newaxis, np.newaxis] * delta
        self.gamma_delta = np.sum(self.normal * delta, axis = (0, 2, 3))
        self.beta_delta = np.sum(delta, axis = (0, 2, 3))
        self.gamma -= self.gamma_delta * lr
        self.beta  -= self.beta_delta * lr
        return input_delta

    def save_model(self):
        return [self.gamma, self.beta]

    def restore_model(self, models):
        self.gamma = models[0]
        self.beta = models[1]

if __name__=="__main__":
    inputs = np.random.rand(1, 2, 3, 3).astype(np.float32)
    affine = True
    num_feature = inputs.shape[1]
    gamma = np.random.rand(num_feature).astype(np.float32)
    beta = np.random.rand(num_feature).astype(np.float32)

    batchnorm = layer_batchnorm(num_feature=num_feature, affine=affine, gamma=gamma, beta=beta)
    output = batchnorm.forward(inputs)
    delta = np.ones(inputs.shape).astype(np.float32)
    partial = batchnorm.backward(delta)

    output_torch, partial_torch, grad_gamma_torch, grad_beta_torch = torch_compare_batchnorm(num_feature, inputs, gamma, beta)
    assert np.mean(np.abs(output - output_torch.cpu().detach().numpy())) < 1e-3, np.mean(np.abs(output - output_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(partial - partial_torch.cpu().detach().numpy())) < 1e-3, np.mean(np.abs(partial - partial_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(batchnorm.gamma_delta - grad_gamma_torch.cpu().detach().numpy())) < 1e-3, np.mean(np.abs(batchnorm.gamma_delta - grad_gamma_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(batchnorm.beta_delta - grad_beta_torch.cpu().detach().numpy())) < 1e-3, np.mean(np.abs(batchnorm.beta_delta - grad_beta_torch.cpu().detach().numpy()))