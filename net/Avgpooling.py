# https://zhuanlan.zhihu.com/p/642043155
import numpy as np
import torch 
from torch import nn
from multiprocessing import Pool, cpu_count, Array, Value

def torch_compare_AvgPool2d(pool_size, strides, padding, inputs):
    network = nn.AvgPool2d(pool_size, strides, padding).requires_grad_(True)
            
    inputs = torch.tensor(inputs, requires_grad=True)
    output = network(inputs)
    sum = torch.sum(output) # make sure the gradient is 1
    kk = sum.backward()
    inputs.retain_grad()
    k = inputs.grad
    return output, k

class avgpooling_layer(object):
    def __init__(self, pool_size, strides=[2,2], padding=[2,2]):
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding  # h ,w

    def pooling_kernel(self, i, j, h, w, pad_input, strides, cnt, ar_output, ar_record):
        h_start = h*strides[0]
        w_start = w*strides[1]
        slide_window = pad_input[i, j, h_start:h_start + pool_size[0], \
                                    w_start:w_start + pool_size[1]]
        maxval = np.max(slide_window)
        index = np.where(slide_window==maxval)
        ar_output[cnt] = maxval
        ar_record[cnt*2] = h_start+index[0][0]
        ar_record[cnt*2 + 1] = w_start+index[1][0]
    
    def im2col(self, ishape):
        ksize = self.pool_size[0] * self.pool_size[1]
        out = np.zeros((ishape[0] * ishape[1] * self.oh * self.ow, ksize))
        cnt = 0
        h_ = []
        w_ = []
        for i in range(ishape[0]):
            for j in range(ishape[1]):
                for h in range(self.oh):
                    h_start = h * self.strides[0]
                    h_.extend([h_start] * self.ow)
                    for w in range(self.ow):
                        w_start = w * self.strides[1]
                        slide_window = self.pad_input[i, j, h_start:h_start + self.pool_size[0], \
                                            w_start:w_start + self.pool_size[1]]
                        kk = slide_window.flatten()
                        out[cnt, :] = kk
                        cnt += 1
                        w_.append(w_start)
        self.record[:, 0] = h_
        self.record[:, 1] = w_
        return out

    def forward(self, inputs):
        self.inputs = inputs
        ishape = self.inputs.shape
        if isinstance(self.padding, str):
            if self.padding=='same':
                self.padding = [self.pool_size[0]//2, self.pool_size[1]//2]
            elif self.padding=='valid':
                self.padding = [0, 0]

        self.oh = (ishape[2] + 2 * self.padding[0] - self.pool_size[0])//self.strides[0] + 1
        self.ow = (ishape[3] + 2 * self.padding[1] - self.pool_size[1])//self.strides[1] + 1
        self.outshape = (ishape[0], ishape[1], self.oh, self.ow)

        if np.sum(self.padding)!=0:
            self.pad_input = np.lib.pad(self.inputs, \
                ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), \
                                mode='constant', constant_values=(0, 0))   
        else:
            self.pad_input = self.inputs

        output = np.zeros(self.outshape)

        self.record = np.zeros((np.prod(output.shape), 2))
        out = self.im2col(ishape)
        meanval = np.mean(out, axis = -1)
        output = np.reshape(meanval, self.outshape)
        self.record = np.array(self.record, dtype = np.int32)
        
        # self.record = []
        # for i in range(ishape[0]):
        #     for j in range(ishape[1]):
        #         for h in range(self.oh):
        #             h_start = h*strides[0]
        #             for w in range(self.ow):
        #                 w_start = w*strides[1]
        #                 slide_window = self.pad_input[i, j, h_start:h_start + self.pool_size[0], \
        #                                     w_start:w_start + self.pool_size[1]]
        #                 maxval = np.mean(slide_window)
        #                 self.record.append([h_start, w_start])
        #                 output[i, j, h, w] = maxval

        #multiprocessing speed up
        # self.record = np.zeros(np.prod(output.shape)*2)
        # out = np.zeros(np.prod(output.shape))
        # cnt = 0
        # P = Pool(cpu_count())
        # for i in range(ishape[0]):
        #     for j in range(ishape[1]):
        #         for h in range(self.oh):
        #             for w in range(self.ow):
        #                 P.apply_async(self.pooling_kernel, args=(i, j, h, w, self.pad_input, strides, cnt, out, self.record,))
        #                 cnt += 1
        # P.close()
        # P.join()
        # self.record = np.reshape(self.record, (-1, 2))
        # output = np.reshape(out, output.shape)

        return output
    
    def backward(self, delta):
        #previous layer delta
        ishape = self.inputs.shape
        input_delta = np.zeros_like(self.pad_input)

        cnt = 0
        windows_size = self.pool_size[0] * self.pool_size[1]
        for i in range(ishape[0]):
            for j in range(ishape[1]):
                for h in range(self.oh):
                    for w in range(self.ow):
                        input_delta[i, j, \
                            self.record[cnt][0]:self.record[cnt][0] + self.pool_size[0], \
                            self.record[cnt][1]:self.record[cnt][1] + self.pool_size[1]] \
                                += delta[i, j, h, w] / windows_size
                        cnt += 1
        #remove padding
        ih = self.pad_input.shape[-2]
        iw = self.pad_input.shape[-1]
        input_delta = input_delta[:, :, self.padding[0]:ih-self.padding[0], self.padding[1]:iw-self.padding[1]]
        return input_delta
    
    def __name__(self):
        return "avgpooling_layer"
        
    def setzero(self):
        pass

    def update(self, lr = 1e-10):
        pass

if __name__=="__main__":
    inputs = np.random.rand(2, 3, 100, 300) * 100
    pool_size = [3, 3]
    strides = [2, 2]
    padding = [1, 1]
    
    avgpooling = avgpooling_layer(pool_size, strides, padding)
    output = avgpooling.forward(inputs)
    delta = np.ones(avgpooling.outshape)
    partial = avgpooling.backward(delta)
    output_torch, partial_torch = torch_compare_AvgPool2d(pool_size, strides, padding, inputs)
    assert np.mean(np.abs(output - output_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(output - output_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(partial - partial_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(partial - partial_torch.cpu().detach().numpy()))