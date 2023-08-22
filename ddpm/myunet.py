import numpy as np
import os
import sys
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
sys.path.append(abspath)

from net.Convolution import convolution_layer
from net.ConvTranspose2d import ConvTranspose2d_layer
from ddpm.channel_position import channel_position_layer
from ddpm.fixposition import fix_postion_layer
from ddpm.myblock import myblock_three_layer, downsample_layer, upsample_layer

class myunet_layer():
    def revise(self):
        self.layers[0].n_steps =66
        self.layers[0].n_steps =20
    
    def rk(self):
        l = self.layers[0]
        l.n_steps = 26
        l.param = np.ones((2, 2))
    
    def mk(self):
        ll = self.layers[0]
        p = ll.param

    def nl(self):
        kk = self.layers[0] 
        k = 0

    def __init__(self, n_steps, embed_dim):
        self.fixposl = fix_postion_layer(n_steps, embed_dim)
        
        # first
        self.cp1 = channel_position_layer(embed_dim, 1)
        self.bl1 = myblock_three_layer(((1, 28, 28), 1, 10), ((10, 28, 28), 10, 10), ((10, 28, 28), 10, 10))
        self.downsample1 = convolution_layer(10, 10, 2*2, 2, 1)
        
        self.cp2 = channel_position_layer(embed_dim, 10)
        self.bl2 = myblock_three_layer(((10, 28//2, 28//2), 10, 20), ((20, 28//2, 28//2), 20, 20), ((20, 28//2, 28//2), 20, 20))
        self.downsample2 = convolution_layer(20, 20, 2*2, 2, 1)

        self.cp3 = channel_position_layer(embed_dim, 20)
        self.bl3 = myblock_three_layer(((20, 7, 7), 20, 20*2), ((20*2, 7, 7), 20*2, 20*2), ((20*2, 7, 7), 20*2, 20*2))
        self.downsample3 = downsample_layer((20*2, 20*2, 2, 1), (20*2, 20*2, 2*2, 2, 1))

        # layer bottleneck
        self.cpmid = channel_position_layer(embed_dim, 20*2)
        self.bl_mid = myblock_three_layer(((20*2, 3, 3), 20*2, 20), ((20, 3, 3), 20, 20), ((20, 3, 3), 20, 20*2))

        # second
        self.up1 = upsample_layer((20*2, 20*2, 2*2, 2, 1), (20*2, 20*2, 2, 1))

        self.cp6 = channel_position_layer(embed_dim, 80)
        self.bl6 = myblock_three_layer(((80, 7, 7), 80, 20*2), ((20*2, 7, 7), 20*2, 20), ((20, 7, 7), 20, 20))
        self.up2 = ConvTranspose2d_layer(20, 20, 2*2, 2, 1)

        self.cpfive = channel_position_layer(embed_dim, 20*2)
        self.blfive = myblock_three_layer(((20*2, 28//2, 28//2), 20*2, 20), ((20, 28//2, 28//2), 20, 10), ((10, 28//2, 28//2), 10, 10))
        self.up3 = ConvTranspose2d_layer(10, 10, 2*2, 2, 1)

        self.cpout = channel_position_layer(embed_dim, 20)
        self.blout = myblock_three_layer(((20, 28, 28), 20, 10), ((10, 28, 28), 10, 10), ((10, 28, 28), 10, 10, False))

        self.outcon = convolution_layer(10, 1, 3, 1, 1)

        k = dir(self)
        self.layers = [self.fixposl, self.cp1, self.bl1, self.downsample1, self.cp2, self.bl2, self.downsample2, self.cp3, self.bl3, self.downsample3, \
                       self.cpmid, self.bl_mid, self.up1, self.cp6, self.bl6, self.up2, self.cpfive, self.blfive, self.up3, self.cpout, self.blout, self.outcon]

    def forward(self, inputs, fix_position):
        fp = self.fixposl.forward(fix_position)
        n = len(inputs)
        
        out1 = self.bl1.forward(inputs + self.cp1.forward(fp).reshape(n, -1, 1, 1))
        out2 = self.bl2.forward(self.downsample1.forward(out1) + self.cp2.forward(fp).reshape(n, -1, 1, 1))
        out3 = self.bl3.forward(self.downsample2.forward(out2) + self.cp3.forward(fp).reshape(n, -1, 1, 1))

        outmid = self.bl_mid.forward(self.downsample3.forward(out3) + self.cpmid.forward(fp).reshape(n, -1, 1, 1))

        outcat3 = np.concatenate([out3, self.up1.forward(outmid)], axis = 1)
        out6 = self.bl6.forward(outcat3 + self.cp6.forward(fp).reshape(n, -1, 1, 1))
        
        outcat2 = np.concatenate([out2, self.up2.forward(out6)], axis = 1)
        outfive = self.blfive.forward(outcat2 + self.cpfive.forward(fp).reshape(n, -1, 1, 1))

        outcat1 = np.concatenate([out1, self.up3.forward(outfive)], axis = 1)
        out = self.blout.forward(outcat1 + self.cpout.forward(fp).reshape(n, -1, 1, 1))

        out = self.outcon.forward(out)

        return out

    def backward(self, delta):
        n = len(delta)
        delta = self.outcon.backward(delta)
        
        outcat1_delta = self.blout.backward(delta)
        _, c, _, _ = outcat1_delta.shape
        cp_delta = np.sum(outcat1_delta, axis = (2, 3)).reshape(n, 1, -1)
        _ = self.cpout.backward(cp_delta)
        out1_delta_up = outcat1_delta[:, :c//2, ...]
        outfive_delta = self.up3.backward(outcat1_delta[:, c//2:, ...])
        
        outcat2_delta = self.blfive.backward(outfive_delta)
        _, c, _, _ = outcat2_delta.shape
        cp_delta = np.sum(outcat2_delta, axis = (2, 3)).reshape(n, 1, -1)
        _ = self.cpfive.backward(cp_delta)
        out2_delta_up = outcat2_delta[:, :c//2, ...]
        out6_delta = self.up2.backward(outcat2_delta[:, c//2:, ...])
        
        outcat3_delta = self.bl6.backward(out6_delta)
        _, c, _, _ = outcat3_delta.shape
        cp_delta = np.sum(outcat3_delta, axis = (2, 3)).reshape(n, 1, -1)
        _ = self.cp6.backward(cp_delta)
        out3_delta_up = outcat3_delta[:, :c//2, ...]
        outmid_delta = self.up1.backward(outcat3_delta[:, c//2:, ...])
        
        midinput_delta = self.bl_mid.backward(outmid_delta)
        out3_delta = self.downsample3.backward(midinput_delta) + out3_delta_up
        cp_delta = np.sum(midinput_delta, axis = (2, 3)).reshape(n, 1, -1)
        _ = self.cpmid.backward(cp_delta)
        
        bl3_input_delta = self.bl3.backward(out3_delta)
        out2_delta = self.downsample2.backward(bl3_input_delta) + out2_delta_up
        cp_delta = np.sum(bl3_input_delta, axis = (2, 3)).reshape(n, 1, -1)
        _ = self.cp3.backward(cp_delta)

        bl2_input_delta = self.bl2.backward(out2_delta)
        out1_delta = self.downsample1.backward(bl2_input_delta) + out1_delta_up
        cp_delta = np.sum(bl2_input_delta, axis = (2, 3)).reshape(n, 1, -1)
        _ = self.cp2.backward(cp_delta)

        input_delta = self.bl1.backward(out1_delta)
        cp_delta = np.sum(input_delta, axis = (2, 3)).reshape(n, 1, -1)
        _ = self.cp1.backward(cp_delta)

        return input_delta

    def update(self, lr):
        for l in self.layers:
            l.update(lr)

    def setzero(self):
        for l in self.layers:
            l.setzero()

    def save_model(self):
        return [l.save_model() for l in self.layers]

    def restore_model(self, models):
        cnt = 0
        for l in self.layers:
            l.restore_model(models[cnt])
            cnt += 1

if __name__=="__main__":
    ml = myunet_layer(1000, 100)
    ml.revise() 
    ml.rk()
    ml.mk()  
    ml.nl()