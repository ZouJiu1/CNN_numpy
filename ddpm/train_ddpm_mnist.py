import os
import sys
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
sys.path.append(abspath)

from net.loss import mean_square_loss
from ddpm.myddpm import myddpm_noise
from ddpm.myunet import myunet_layer
import numpy as np
import pickle
import imageio

from torchvision import datasets
import pandas as pd

# https://en.wikipedia.org/wiki/AlexNet
# https://pytorch.org/vision/stable/_modules/torchvision/models/alexnet.html#alexnet
# https://github.com/l5shi/Image-Recognition-on-MNIST-dataset/blob/master/AlexNet.ipynb

def genimage(myunet: myunet_layer, noisegen: myddpm_noise, gifname : str, frame_pergif = 100, n_sample = 16, c = 1, h = 30 -2, w = 30 -2):
    frame_id = np.linspace(0, noisegen.n_steps, frame_pergif).astype(np.uint)
    frames = []

    inputs = np.random.rand(n_sample, c, h, w)
    for id, fixposition in enumerate(np.arange(noisegen.n_steps)[::-1]):
        array_fixpos = np.ones((n_sample, 1), dtype=np.uint) * fixposition
        predict_noisy = myunet.forward(inputs, array_fixpos)
        fix_alpha = noisegen.alpha[fixposition]
        fixprod_alpha = noisegen.prod_alpha[fixposition]
        
        inputs = (1 / np.sqrt(fix_alpha)) * (inputs - (1 - fix_alpha) / np.sqrt(1 - fixprod_alpha) * predict_noisy)

        if fixposition > 0:
            z = np.random.rand(n_sample, c, h, w)
            
            #0
            fixbeta = noisegen.beta[fixposition]
            fix_sigma = np.sqrt(fixbeta)
            
            #1
            # preprod_alpha = noisegen.prod_alpha[fixposition - 1] if fixposition > 0 else noisegen.prod_alpha[0]
            # prebeta = ((1 - preprod_alpha) / (1 - fixprod_alpha)) * fixbeta
            # fix_sigma = np.sqrt(prebeta)
            
            inputs = inputs + fix_sigma * z
        
        if id in frame_id or fixposition==0:
            normalized = inputs.copy()
            minval = np.min(normalized, axis = (1, 2, 3), keepdims=True)
            maxval = np.max(normalized, axis = (1, 2, 3), keepdims=True)
            normalized = ((normalized - minval) / maxval) * 255

            n, c, h, w = normalized.shape
            normalized = np.transpose(normalized, (0, 2, 3, 1)).astype(np.uint8)

            num = int(np.sqrt(n))
            width = num * h
            height = num * w
            gen_img = np.zeros((height, width, c), dtype = np.uint8)
            
            for i in range(len(normalized)):
                jh = i // num
                jw = i % num
                gen_img[h*jh:h*(jh+1), w*jw:w*(jw+1), :] = normalized[i, ...]
            
            frames.append(gen_img)

    with imageio.get_writer(gifname, mode="I") as obj:
        for id, frame in enumerate(frames):
            obj.append_data(frame)
            if id == len(frames) - 1:
                for _ in range(frame_pergif // 3):
                    obj.append_data(frames[-1])
    return inputs

def train_ddpm():
    n_steps = 3
    embed_dim  = 100
    display = True
    myunet = myunet_layer(n_steps, embed_dim)
    noisegen = myddpm_noise(n_steps, embed_dim)
    
    if os.path.exists(pretrained_model):
        with open(pretrained_model, 'rb') as obj:
            models = pickle.load(obj)
            noisegen.restore_model(models)
            del models

    epoch = 200
    batchsize = 1
    lr = 0.001 / batchsize
    datapath = os.path.join(abspath, 'dataset')
    os.makedirs(datapath, exist_ok=True)
    modelpath = os.path.join(abspath, 'model')
    os.makedirs(modelpath, exist_ok=True)

    datatest = datasets.MNIST(root = datapath, train=False, download=True)
    datatrain = datasets.MNIST(root = datapath, train=True, download=True)
    testdata, testlabel = datatest._load_data()
    datas, labels = datatrain._load_data()
    # */255
    testdata, testlabel = testdata.cpu().numpy() / 255, testlabel.cpu().numpy()
    datas, labels = datas.cpu().numpy() / 255, labels.cpu().numpy()
    datas = np.concatenate([datas, testdata], axis = 0)
    datas = (datas - (1/2.0)) * 2
    datas = datas[:3, :, :]
    del testlabel, labels, testdata

    number_image = datas.shape[0]
    loss = 999999
    iters = number_image//batchsize # + number_image%batchsize

    start_epoch = 0
    if os.path.exists(pretrained_model):
        start_epoch = int(pretrained_model.split(os.sep)[-1].split("_")[1])

    alliter = 0
    for i in range(start_epoch, epoch):
        meanloss = 0
        # if i!=0:
            # lr = lr * dot
        for j in range(iters):
            alliter += 1
            images = datas[j*batchsize:(j+1)*batchsize, :, :]
            images = images[:, np.newaxis, :, :]
            images = images
            
            n_, c_, h_, w_ = images.shape
            noisy = np.random.rand(n_, c_, h_, w_)
            fixposition = np.random.randint(0, n_steps, (n_))

            noisy = np.ones((n_, c_, h_, w_))
            fixposition = np.random.randint(0, 1, (n_,))

            noisy_image = noisegen.forward(images, fixposition, noisy)
            predict_noisy = myunet.forward(noisy_image, fixposition.reshape(n_, -1))

            loss, delta = mean_square_loss(noisy, predict_noisy)
            # loss = loss * noisy.size
            # delta = delta * noisy.size
            delta = myunet.backward(delta)
            myunet.update(lr)
            myunet.setzero()

            meanloss += loss
            fpwrite.write("epoch:{}, lr: {:.6f}, loss: {:.6f}, iters: {}\n".format(i, lr, loss, j, ))
            fpwrite.flush()

        if i == epoch - (2//2):
            genimage(myunet, noisegen, os.path.join(gifdir, str(i)+".gif"), n_sample = 100)

        meanloss = meanloss / iters
        # # savemodel
        # allmodel = myunet.save_model()
        # name = "epoch_"+str(i)+"_loss_"+str(round(meanloss, 6))+".pkl"

        # with open(os.path.join(modelpath, name), 'wb') as obj:
        #     pickle.dump(allmodel, obj)

        fpwrite.write("epoch: {}\n\n".format(i))
        fpwrite.flush()

if __name__ =="__main__":
    savepath = abspath
    pretrained_model = r''
    gifdir = os.path.join(savepath, 'ddpm')
    logdir = os.path.join(savepath, 'log')
    logfile = os.path.join(logdir, 'logddpm.txt')
    fpwrite = open(logfile, 'w', encoding='utf-8')
    
    train_ddpm()
    fpwrite.close()