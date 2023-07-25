# https://zhuanlan.zhihu.com/p/642043155
import numpy as np
import torch 
from torch import nn
from copy import deepcopy

def torch_compare_ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias, inputs, params, bias_params, delta, output_padding):
    network = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, output_padding=output_padding, \
        bias = bias).requires_grad_(True)
    cnt = 0
    # params = np.transpose(params, (1, 0, 2, 3))
    for i in network.parameters():
        if cnt==0:
            i.data = torch.from_numpy(params)
            i.retain_grad = True
        else:
            i.data = torch.from_numpy(bias_params)
            i.retain_grad = True
        cnt += 1
            
    inputs = torch.tensor(inputs, requires_grad=True)
    output = network(inputs)
    # sum = torch.sum(output) # make sure the gradient is 1
    output.backward(torch.tensor(delta))
    grad_params = 0
    grad_bias   = 0
    cnt = 0
    for i in network.parameters():
        if cnt==0:
            grad_params = i.grad
        else:
            grad_bias = i.grad
        cnt += 1
    inputs.retain_grad()
    k = inputs.grad
    return output, k, grad_params, grad_bias

class ConvTranspose2d_layer(object):
    def __init__(self, in_channel, out_channel, kernel_size, output_padding=0, stride=[1,1], padding=[0,0], bias=False, params=[], bias_params=[]):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.output_padding = output_padding
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        self.kernel_size = kernel_size
        self.bias = bias
        if list(params)!=[]:
            self.params = params
        else:
            ranges = np.sqrt(6 / (in_channel + out_channel))
            self.params = np.random.uniform(-ranges, ranges, (out_channel, in_channel, kernel_size[0], kernel_size[1]))

        if bias and list(bias_params)!=[]:
            self.bias_params = bias_params
        else:
            ranges = np.sqrt(6 / (in_channel + out_channel))
            self.bias_params = np.random.uniform(-ranges, ranges, (out_channel))

        self.params_delta = np.zeros((in_channel, out_channel, kernel_size[0], kernel_size[1])).astype(np.float64)
        self.bias_delta = np.zeros(out_channel).astype(np.float64)
        if isinstance(stride, int):
            stride = [stride, stride]
        self.stride = stride
        if isinstance(padding, int):
            padding = [padding, padding]
        self.padding = padding

    def im2col(self, kernel_size, outshape, inchannel, pad_input, stride):
        im_col = np.zeros((outshape[0] * outshape[2] * outshape[3], \
                        np.prod(kernel_size) * inchannel)).astype(np.float64) # (ob * oh * ow, kw*kh*ic)
        cnt = 0
        for i in range(outshape[0]):
            for h in range(outshape[2]):
                h_start = h * stride[0]
                for w in range(outshape[3]):
                    w_start = w * stride[1]
                    kernel_size_channel = []
                    for j in range(inchannel): # in_channel
                        slide_window = pad_input[i, j, h_start:h_start + kernel_size[0], \
                                            w_start:w_start + kernel_size[1]]
                        flatten = slide_window.flatten()
                        kernel_size_channel.extend(flatten)
                    im_col[cnt, :] = kernel_size_channel
                    cnt += 1
        return im_col

    def common_backward_calcul(self, delta):
        input_delta = np.zeros(self.ishape).astype(np.float64)
        delta = np.lib.pad(delta, \
                ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), \
                                mode='constant', constant_values=(0, 0))
        if self.bias:
            self.bias_delta += np.sum(delta, axis=(0, 2, 3))
        for i in range(self.ishape[0]):
            for ic in range(self.ishape[1]):
                for h in range(self.ishape[2]):
                    h_start = h*self.stride[0]
                    for w in range(self.ishape[3]):
                        w_start = w*self.stride[1]
                        val = self.inputs[i, ic, h, w]
                        sum = 0
                        for j in range(self.outshape[1]):
                            slide_window = delta[i, j, h_start:h_start + self.kernel_size[0], \
                                                w_start:w_start + self.kernel_size[1]]
                            sum += np.sum(slide_window * self.params[ic, j, :, :])
                            self.params_delta[ic, j, :, :] += slide_window * val
                        input_delta[i, ic, h, w] = sum
        return input_delta

    def backward(self, delta):
        delta = np.lib.pad(delta, \
                ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), \
                                mode='constant', constant_values=(0, 0))
        if self.bias:
            self.bias_delta += np.sum(delta, axis=(0, 2, 3))

        im_col = self.im2col(self.kernel_size, self.ishape, self.outshape[1], delta, self.stride)
        kernel_col = np.reshape(self.params, (self.params.shape[0], -1)).T
        # (ob * oh * ow, kw*kh*ic) * (kw*kh*ic, oc)    =   (ob * oh * ow, oc)
        input_delta = np.matmul(im_col, kernel_col) 
        input_delta = np.reshape(input_delta, (self.ishape[0], self.ishape[2], self.ishape[3], self.ishape[1]))
        input_delta = np.transpose(input_delta, (0, 3, 1, 2))

        N_in, C_in, H_in, W_in = self.ishape
        S_h, S_w = self.stride
        # internel pad inputs
        internal_pad_H = (H_in - 1) * (S_h - 1) + H_in  #IPH
        internal_pad_W = (W_in - 1) * (S_w - 1) + W_in  #IPW
        pad_inputs = np.zeros((N_in, C_in, internal_pad_H, internal_pad_W))
        pad_inputs[:, :, ::S_h, ::S_w] = self.inputs

        #calcul params gradient
        H_outshape_params_gradient = (self.outshape[2] + 2 * self.padding[0] - internal_pad_H) + 1  #OH
        W_outshape_params_gradient = (self.outshape[3] + 2 * self.padding[1] - internal_pad_W) + 1  #OW
        outshape_params_gradient = (self.outshape[1], self.ishape[1], H_outshape_params_gradient, \
                                    W_outshape_params_gradient) # ic oc OH OW
        kernel_params_gradient = (internal_pad_H, internal_pad_W)
        pad_delta_T = np.transpose(delta, (1, 0, 2, 3)) #(oc, N, ih, iw)
        im_col = self.im2col(kernel_params_gradient, outshape_params_gradient, pad_delta_T.shape[1], pad_delta_T, [1, 1])
        pad_input_T = np.transpose(pad_inputs, (1, 0, 2, 3)) #(ic, N, IPH, IPW)
        kernel_col = np.reshape(pad_input_T, (pad_input_T.shape[0], -1)).T
        # (oc * OH * OW, IPH * IPH * N) * (IPH * IPH * N, ic) = (oc * OH * OW, ic)
        output = np.matmul(im_col, kernel_col)
        output = np.reshape(output, (outshape_params_gradient[0], \
            outshape_params_gradient[2], outshape_params_gradient[3], outshape_params_gradient[1]))
        params_delta = np.transpose(output, (3, 0, 1, 2))
        if H_outshape_params_gradient > self.kernel_size[0]:
            self.params_delta += params_delta[:, :, :self.kernel_size[0], :]
        if W_outshape_params_gradient > self.kernel_size[1]:
            self.params_delta += params_delta[:, :, :, :self.kernel_size[1]]
        return input_delta

    def setzero(self):
        self.params_delta[...]  = 0.0
        self.bias_delta[...] = 0.0

    def update(self, lr = 1e-10):
        self.params -= self.params_delta * lr
        if self.bias:
            self.bias_params   -= self.bias_delta * lr

    def save_model(self):
        return [self.params, self.bias_params]

    def restore_model(self, models):
        self.params = models[0]
        self.bias_params = models[1]

    def __name__(self):
        return "ConvTranspose2d_layer"

    def forward_common(self, inputs):
        # previous layer delta
        self.inputs = inputs
        self.ishape = inputs.shape
        self.oh = (self.ishape[2] - 1) * self.stride[0] + self.kernel_size[0] - 2 * self.padding[0] + self.output_padding[0]  # convert (self.ishape[2] + 2 * self.padding[0] - self.kernel_size[0])//self.stride[0] + 1
        self.ow = (self.ishape[3] - 1) * self.stride[1] + self.kernel_size[1] - 2 * self.padding[1] + self.output_padding[1]  # (self.ishape[3] + 2 * self.padding[1] - self.kernel_size[1])//self.stride[1] + 1
        now_outshape = (self.ishape[0], self.out_channel, self.oh + 2 * self.padding[0] - self.output_padding[0], \
                        self.ow + 2 * self.padding[1] - self.output_padding[1])
        self.outshape = (self.ishape[0], self.out_channel, self.oh, self.ow)

        outputs = np.zeros(now_outshape).astype(np.float64)
        for i in range(self.ishape[0]):
            for ic in range(self.ishape[1]):
                for h in range(self.ishape[2]):
                    h_start = h*self.stride[0]
                    for w in range(self.ishape[3]):
                        inputs_val = inputs[i, ic, h, w]
                        w_start = w*self.stride[1]
                        for j in range(self.outshape[1]):
                            outputs[i, j, h_start:h_start + self.kernel_size[0], \
                                w_start:w_start + self.kernel_size[1]] += self.params[ic, j, :, :] * inputs_val
        ih = outputs.shape[-2]
        iw = outputs.shape[-1]
        outputs = outputs[:, :, self.padding[0]:ih-self.padding[0] + self.output_padding[0], \
            self.padding[1]:iw-self.padding[1] + self.output_padding[1]]

        if self.bias:
            outputs = outputs + self.bias_params[np.newaxis, :, np.newaxis, np.newaxis]

        return outputs

    def forward(self, inputs):
        # previous layer delta convert of convolution
        self.inputs = inputs
        self.ishape = inputs.shape
        self.oh = (self.ishape[2] - 1) * self.stride[0] + self.kernel_size[0] - 2 * self.padding[0] + self.output_padding[0]  # convert (self.ishape[2] + 2 * self.padding[0] - self.kernel_size[0])//self.stride[0] + 1
        self.ow = (self.ishape[3] - 1) * self.stride[1] + self.kernel_size[1] - 2 * self.padding[1] + self.output_padding[1]  # (self.ishape[3] + 2 * self.padding[1] - self.kernel_size[1])//self.stride[1] + 1
        self.outshape = (self.ishape[0], self.out_channel, self.oh, self.ow)

        N_in, C_in, H_in, W_in = self.ishape
        S_h, S_w = self.stride
        
        # internel pad inputs
        internal_pad_H = (H_in - 1) * (S_h - 1) + H_in  #IPH
        internal_pad_W = (W_in - 1) * (S_w - 1) + W_in  #IPW
        pad_inputs = np.zeros((N_in, C_in, internal_pad_H, internal_pad_W))
        pad_inputs[:, :, ::S_h, ::S_w] = inputs

        # calcul externel pad shape and outputs
        remain_h = (self.oh + 2 * self.padding[0] - self.kernel_size[0]) % self.stride[0]
        remain_w = (self.ow + 2 * self.padding[1] - self.kernel_size[1]) % self.stride[1]
        pad_top = self.kernel_size[0] - 1 - self.padding[0]
        pad_bottom = self.kernel_size[0] - 1 - self.padding[0] + remain_h + self.output_padding[0]
        pad_left = self.kernel_size[1] - 1 - self.padding[1]
        pad_right = self.kernel_size[1] - 1 - self.padding[1] + remain_w + self.output_padding[1]
        pad_inputs_external = np.lib.pad(pad_inputs, \
                ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), \
                                mode='constant', constant_values=(0, 0)) #(N, oc, EPH, EPW)
        #rotate 180
        cp_params = deepcopy(self.params)
        # cp_params = np.rot90(cp_params, 2, axes=(2, 3))
        cp_params = np.flip(np.flip(cp_params, 2), 3)
        
        params_T = np.transpose(cp_params, (1, 0, 2, 3)) #(oc, ic, kh, kw)
        im_col = self.im2col((params_T.shape[2], params_T.shape[3]), self.outshape, params_T.shape[1], pad_inputs_external, [1, 1])
        kernel_col = np.reshape(params_T, (params_T.shape[0], -1)).T
        # (N * oh * ow, kw * kh * ic) * (kw * kh * ic, oc) = (N * oh * ow, oc)
        output = np.matmul(im_col, kernel_col)
        output = np.reshape(output, (self.outshape[0], self.outshape[2], self.outshape[3], self.outshape[1]))
        outputs = np.transpose(output, (0, 3, 1, 2))
        
        if self.bias:
            outputs = outputs + self.bias_params[np.newaxis, :, np.newaxis, np.newaxis]

        return outputs

def train_single():
    out_channel = 6
    outputs = np.random.rand(2, out_channel, 29, 29).astype(np.float64)
    inputs = np.random.rand(2, 10, 10, 10).astype(np.float64)
    batchsize = inputs.shape[0]
    in_channel = inputs.shape[1]
    ih = inputs.shape[2]
    iw = inputs.shape[3]
    kernel_size = [3, 3]
    stride = [3, 3]
    padding = [1, 1]
    output_padding = [1, 1]
    bias = True
    params = np.random.standard_normal((in_channel, out_channel, kernel_size[0], kernel_size[1])) / np.sqrt(in_channel/2) / 10
    params = params.astype(np.float64)
    bias_params = None
    if bias:
        bias_params = np.random.standard_normal(out_channel) / np.sqrt(in_channel/2)
        bias_params = bias_params.astype(np.float64)
    ConvTranspose2d = ConvTranspose2d_layer(in_channel, out_channel, kernel_size, output_padding, stride, padding, bias, params.copy(), bias_params.copy())
    for i in range(30000):
        out = ConvTranspose2d.forward(inputs)
        sum = np.sum((outputs - out) * (outputs - out))
        delta = 2*(out - outputs)
        # partial_, = convolution.backward_common(delta, 0.0001)
        partial = ConvTranspose2d.backward(delta, 0.0001)
        print(sum)

if __name__=="__main__":
    # train_single()

    out_channel = 6
    inputs = np.random.rand(2, 10, 10, 30).astype(np.float64)
    batchsize = inputs.shape[0]
    in_channel = inputs.shape[1]
    ih = inputs.shape[2]
    iw = inputs.shape[3]
    
    kernel_size = [3, 3]
    stride = [3, 3]
    padding = [1, 1]
    output_padding = [1, 1]
    bias = True
    params = np.random.standard_normal((in_channel, out_channel, kernel_size[0], kernel_size[1])) / np.sqrt(in_channel/2) / 10
    params = params.astype(np.float64)
    if bias:
        bias_params = np.random.standard_normal(out_channel) / np.sqrt(in_channel/2)
        bias_params = bias_params.astype(np.float64)

    convolution = ConvTranspose2d_layer(in_channel, out_channel, kernel_size, output_padding, stride, padding, bias, params.copy(), bias_params.copy())
    output = convolution.forward(inputs)
    output_k = convolution.forward_common(inputs)
    delta = np.ones(convolution.outshape).astype(np.float64)
    partial_k = convolution.common_backward_calcul(delta)
    partial = convolution.backward(delta)

    output_torch, partial_torch, grad_params_torch, grad_bias_torch = torch_compare_ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias, inputs, params.copy(), bias_params.copy(), delta, output_padding)
    assert np.mean(np.abs(output - output_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(output - output_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(partial - partial_torch.cpu().detach().numpy())) < 1e-3, np.mean(np.abs(partial - partial_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(convolution.params_delta - grad_params_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(convolution.params_delta - grad_params_torch.cpu().detach().numpy()))
    if bias:
        assert np.mean(np.abs(convolution.bias_delta - grad_bias_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(convolution.bias_delta - grad_bias_torch.cpu().detach().numpy()))