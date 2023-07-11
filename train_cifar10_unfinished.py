import os
from dropout import dropout_layer
from Convolution import convolution_layer
from MaxPool2d import maxpooling_layer
from Avgpooling import avgpooling_layer
from loss import cross_entropy_loss, mean_square_loss
from fullconnect import fclayer
from activation import ReLU
from flatten import flatten_layer
import numpy as np

from torchvision import datasets
from PIL import Image

abspath = os.path.abspath(__file__)
filename = abspath.split(os.sep)[-1]
abspath = abspath.replace(filename, "")

# https://en.wikipedia.org/wiki/AlexNet
# https://pytorch.org/vision/stable/_modules/torchvision/models/alexnet.html#alexnet
# https://github.com/l5shi/Image-Recognition-on-MNIST-dataset/blob/master/AlexNet.ipynb
def Alexnet_train(num_classes):
    convolu0 = convolution_layer(1, 32, kernel_size=3, stride=1, padding=1)
    relu0    = ReLU()
    max0     = maxpooling_layer(pool_size=3, strides=2)
    convolu1 = convolution_layer(32, 60, kernel_size=3, padding=2)
    relu1    = ReLU()
    max1     = maxpooling_layer(pool_size=3, strides=2)
    convolu2 = convolution_layer(60, 96, kernel_size=3, padding=1)
    relu2    = ReLU()
    convolu3 = convolution_layer(96, 60, kernel_size=3, padding=1)
    relu3    = ReLU()
    convolu4 = convolution_layer(60, 60, kernel_size=3, padding=1)
    relu4    = ReLU()
    max2     = maxpooling_layer(pool_size=3, strides=2)
    # dropout0 = dropout_layer(dropout_probability=0.5)
    fla0     = flatten_layer()
    fc0      = fclayer(1080//2, 1000)
    relu5    = ReLU()
    # dropout1 = dropout_layer(dropout_probability=0.5)
    fc1      = fclayer(1000, 600)
    relu6    = ReLU()
    fc2      = fclayer(600, num_classes)
    # layers = [convolu0, relu0, max0, convolu1, relu1, max1, convolu2, relu2, convolu3, relu3, \
    #           convolu4, relu4, max2, dropout0, fla0, fc0, relu5, dropout1, fc1, relu6, fc2]
    layers = [convolu0, relu0, max0, convolu1, relu1, max1, convolu2, relu2, convolu3, relu3, \
              convolu4, relu4, max2, fla0, fc0, relu5, fc1, relu6, fc2]
    
    epoch = 100
    batchsize = 10
    lr = 0.00001
    datapath = os.path.join(abspath, 'dataset')
    os.makedirs(datapath, exist_ok=True)
    datatest = datasets.MNIST(root = datapath, train=False, download=True)
    datatrain = datasets.MNIST(root = datapath, train=True, download=True)
    testdata, testlabel = datatest._load_data()
    datas, labels = datatrain._load_data()
    number_image = datas.shape[0]
    # for i in range(number_image):
    #     img = datas[i, :, :]
    #     Image.fromarray(img.cpu().numpy()).save(os.path.join(abspath, 'dataset', str(i) + ".jpg"))
    iters = number_image//batchsize + number_image%batchsize
    for i in range(epoch):
        for j in range(iters):
            label = np.zeros((batchsize, 10))
            images = datas[j*batchsize:(j+1)*batchsize, :, :]
            label_single = labels[j*batchsize:(j+1)*batchsize]
            label[range(batchsize), label_single] = 1
            images = images[:, np.newaxis, :, :]
            for l in range(len(layers)):
                images = layers[l].forward(images)
            loss, delta, predict = cross_entropy_loss(images, label)
            p = np.argmax(predict, axis=-1)
            precision = np.sum(label_single==p)/len(label_single)
                
            fpwrite.write("epoch:{}, lr: {:.6f}, loss: {:.6f}, iters: {}, precision: {:.6f}\n".format(i, lr, loss, j, precision))
            fpwrite.flush()
            for l in range(len(layers)-1, -1, -1):
                delta = layers[l].backward(delta, lr)
        acc = 0
        length = 0
        for j in range(len(testdata)):
            images = testdata[j*batchsize:(j+1)*batchsize, :, :]
            images = images[:, np.newaxis, :, :]
            label_single = testlabel[j*batchsize:(j+1)*batchsize]
            for l in range(len(layers)):
                images = layers[l].forward(images)
            loss, delta, predict = cross_entropy_loss(images, label)
            p = np.argmax(predict, axis=-1)
            length += len(label_single)
            acc += np.sum(label_single==p)
        precision = acc / length
        fpwrite.write("epoch: {}, testset precision: {}\n\n".format(i, precision))
        fpwrite.flush()

if __name__ =="__main__":
    savepath = abspath
    logdir = os.path.join(savepath, 'log')
    logfile = os.path.join(logdir, 'log_alexnet_cifar10.txt')
    fpwrite = open(logfile, 'w', encoding='utf-8')
    
    Alexnet_train(10)
    fpwrite.close()