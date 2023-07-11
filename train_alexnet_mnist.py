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
import pickle
from layerbatchnorm import layer_batchnorm

from torchvision import datasets
from PIL import Image
import pandas as pd

abspath = os.path.abspath(__file__)
filename = abspath.split(os.sep)[-1]
abspath = abspath.replace(filename, "")

# https://en.wikipedia.org/wiki/AlexNet
# https://pytorch.org/vision/stable/_modules/torchvision/models/alexnet.html#alexnet
# https://github.com/l5shi/Image-Recognition-on-MNIST-dataset/blob/master/AlexNet.ipynb

def Alexnet_train(num_classes):
    if choose=="morelarge":
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
        convolu5 = convolution_layer(60, 60, kernel_size=3, padding=1)
        relu5    = ReLU()
        convolu6 = convolution_layer(60, 130, kernel_size=3, padding=1)
        relu6    = ReLU()
        convolu7 = convolution_layer(130, 60, kernel_size=3, padding=1)
        relu7    = ReLU()
        convolu8 = convolution_layer(60, 60, kernel_size=3, padding=1)
        relu8    = ReLU()
        max2     = maxpooling_layer(pool_size=3, strides=2)
        dropout0 = dropout_layer(dropout_probability=0.3)
        fla0     = flatten_layer()
        fc0      = fclayer(1080//2, 1000)
        relu9    = ReLU()
        dropout1 = dropout_layer(dropout_probability=0.3)
        fc1      = fclayer(1000, 600)
        relu10    = ReLU()
        fc2      = fclayer(600, num_classes)
        layers = [convolu0, relu0, max0, convolu1, relu1, max1, convolu2, relu2, convolu3, relu3, \
                  convolu4, relu4, convolu5, relu5, convolu6, relu6, convolu7, relu7, convolu8, relu8, \
                  max2, fla0, fc0, relu9, fc1, relu10, fc2]
    elif choose=="large":
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
        dropout0 = dropout_layer(dropout_probability=0.3)
        fla0     = flatten_layer()
        fc0      = fclayer(1080//2, 1000)
        relu5    = ReLU()
        dropout1 = dropout_layer(dropout_probability=0.3)
        fc1      = fclayer(1000, 600)
        relu6    = ReLU()
        fc2      = fclayer(600, num_classes)
        layers = [convolu0, relu0, max0, convolu1, relu1, max1, convolu2, relu2, convolu3, relu3, \
                convolu4, relu4, max2, fla0, fc0, relu5, fc1, relu6, fc2]
    elif choose=="large_bn":
        convolu0 = convolution_layer(1, 32, kernel_size=3, stride=1, padding=1)
        bn0      = layer_batchnorm(32)
        relu0    = ReLU()
        max0     = maxpooling_layer(pool_size=3, strides=2)
        convolu1 = convolution_layer(32, 60, kernel_size=3, padding=2)
        bn1      = layer_batchnorm(60)
        relu1    = ReLU()
        max1     = maxpooling_layer(pool_size=3, strides=2)
        convolu2 = convolution_layer(60, 96, kernel_size=3, padding=1)
        bn2      = layer_batchnorm(96)
        relu2    = ReLU()
        convolu3 = convolution_layer(96, 60, kernel_size=3, padding=1)
        bn3      = layer_batchnorm(60)
        relu3    = ReLU()
        convolu4 = convolution_layer(60, 60, kernel_size=3, padding=1)
        bn4      = layer_batchnorm(60)
        relu4    = ReLU()
        max2     = maxpooling_layer(pool_size=3, strides=2)
        dropout0 = dropout_layer(dropout_probability=0.3)
        fla0     = flatten_layer()
        fc0      = fclayer(1080//2, 1000)
        relu5    = ReLU()
        dropout1 = dropout_layer(dropout_probability=0.3)
        fc1      = fclayer(1000, 600)
        relu6    = ReLU()
        fc2      = fclayer(600, num_classes)
        layers = [convolu0, bn0, relu0, max0, convolu1, bn1, relu1, max1, convolu2, bn2, relu2, convolu3, bn3, relu3, \
                convolu4, bn4, relu4, max2, fla0, fc0, relu5, fc1, relu6, fc2]
    elif choose=='mid':
        convolu0 = convolution_layer(1, 10, kernel_size=3, stride=1, padding=1)
        relu0    = ReLU()
        max0     = maxpooling_layer(pool_size=3, strides=2)
        convolu1 = convolution_layer(10, 20, kernel_size=3, padding=2)
        relu1    = ReLU()
        max1     = maxpooling_layer(pool_size=3, strides=2)
        convolu2 = convolution_layer(20, 20, kernel_size=3, padding=1)
        relu2    = ReLU()
        convolu3 = convolution_layer(20, 30, kernel_size=3, padding=1)
        relu3    = ReLU()
        convolu4 = convolution_layer(30, 30, kernel_size=3, padding=1)
        relu4    = ReLU()
        max2     = maxpooling_layer(pool_size=3, strides=2)
        dropout0 = dropout_layer(dropout_probability=0.3)
        fla0     = flatten_layer()
        fc0      = fclayer(270, 300)
        relu5    = ReLU()
        dropout1 = dropout_layer(dropout_probability=0.3)
        fc1      = fclayer(300, 100)
        relu6    = ReLU()
        fc2      = fclayer(100, num_classes)
        layers = [convolu0, relu0, max0, convolu1, relu1, max1, convolu2, relu2, convolu3, relu3, \
                convolu4, relu4, max2, fla0, fc0, relu5, fc1, relu6, fc2]
    elif choose=='small':
        convolu0 = convolution_layer(1, 6, kernel_size=3, stride=1, padding=1)
        relu0    = ReLU()
        max0     = maxpooling_layer(pool_size=3, strides=2)
        convolu1 = convolution_layer(6, 10, kernel_size=3, padding=2)
        relu1    = ReLU()
        max1     = maxpooling_layer(pool_size=3, strides=2)
        convolu2 = convolution_layer(10, 10, kernel_size=3, padding=1)
        relu2    = ReLU()
        convolu3 = convolution_layer(10, 20, kernel_size=3, padding=1)
        relu3    = ReLU()
        convolu4 = convolution_layer(20, 20, kernel_size=3, padding=1)
        relu4    = ReLU()
        max2     = maxpooling_layer(pool_size=3, strides=2)
        dropout0 = dropout_layer(dropout_probability=0.3)
        fla0     = flatten_layer()
        fc0      = fclayer(180, 180)
        relu5    = ReLU()
        dropout1 = dropout_layer(dropout_probability=0.3)
        fc1      = fclayer(180, 90)
        relu6    = ReLU()
        fc2      = fclayer(90, num_classes)
        # layers = [convolu0, relu0, max0, convolu1, relu1, max1, convolu2, relu2, convolu3, relu3, \
        #           convolu4, relu4, max2, dropout0, fla0, fc0, relu5, dropout1, fc1, relu6, fc2]
        layers = [convolu0, relu0, max0, convolu1, relu1, max1, convolu2, relu2, convolu3, relu3, \
                convolu4, relu4, max2, fla0, fc0, relu5, fc1, relu6, fc2]
    elif choose=='small_bn':
        convolu0 = convolution_layer(1, 6, kernel_size=3, stride=1, padding=1)
        bn0      = layer_batchnorm(6)
        relu0    = ReLU()
        max0     = maxpooling_layer(pool_size=3, strides=2)
        convolu1 = convolution_layer(6, 10, kernel_size=3, padding=2)
        bn1      = layer_batchnorm(10)
        relu1    = ReLU()
        max1     = maxpooling_layer(pool_size=3, strides=2)
        convolu2 = convolution_layer(10, 10, kernel_size=3, padding=1)
        bn2      = layer_batchnorm(10)
        relu2    = ReLU()
        convolu3 = convolution_layer(10, 20, kernel_size=3, padding=1)
        bn3      = layer_batchnorm(20)
        relu3    = ReLU()
        convolu4 = convolution_layer(20, 20, kernel_size=3, padding=1)
        bn4      = layer_batchnorm(20)
        relu4    = ReLU()
        max2     = maxpooling_layer(pool_size=3, strides=2)
        dropout0 = dropout_layer(dropout_probability=0.3)
        fla0     = flatten_layer()
        fc0      = fclayer(180, 180)
        relu5    = ReLU()
        dropout1 = dropout_layer(dropout_probability=0.3)
        fc1      = fclayer(180, 90)
        relu6    = ReLU()
        fc2      = fclayer(90, num_classes)
        layers = [convolu0, bn0, relu0, max0, convolu1, bn1, relu1, max1, convolu2, bn2, relu2, convolu3, bn3, relu3, \
                convolu4, bn4, relu4, max2, fla0, fc0, relu5, fc1, relu6, fc2]
    
    if os.path.exists(pretrained_model):
        with open(pretrained_model, 'rb') as obj:
            models = pickle.load(obj)
        cnt = 0
        for l in layers:
            k = dir(l)
            if 'restore_model' in k and 'save_model' in k:
                l.restore_model(models[cnt])
                cnt += 1
            
    epoch = 20
    batchsize = 100
    lr = 0.0001
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
    #one-hot
    test_label = np.zeros((len(testlabel), 10))
    test_label[range(len(testlabel)), testlabel] = 1
    test_l = testlabel.copy()
    testlabel = test_label.copy()
    train_label = np.zeros((len(labels), 10))
    train_label[range(len(labels)), labels] = 1
    train_l = labels.copy()
    labels = train_label.copy()
    del test_label, train_label

    number_image = datas.shape[0]
    # for i in range(number_image):
    #     img = datas[i, :, :]
    #     Image.fromarray(img.cpu().numpy()).save(os.path.join(abspath, 'dataset', str(i) + ".jpg"))
    loss = 999999
    iters = number_image//batchsize + number_image%batchsize
    dot = np.power(0.001, 1/epoch)
    for i in range(epoch):
        meanloss = 0
        # if i!=0:
            # lr = lr * dot
        k = np.arange(len(train_l))
        np.random.shuffle(k)
        datas = datas[k]
        labels = labels[k]
        
        train_l = train_l[k]
        for j in range(iters):
            images = datas[j*batchsize:(j+1)*batchsize, :, :]
            label = labels[j*batchsize:(j+1)*batchsize, :]
            label_single = train_l[j*batchsize:(j+1)*batchsize]
            images = images[:, np.newaxis, :, :]
            images = images
            for l in range(len(layers)):
                images = layers[l].forward(images)
            loss, delta, predict = cross_entropy_loss(images, label)
            meanloss += loss
            p = np.argmax(predict, axis=-1)
            precision = np.sum(label_single==p) / len(label_single)
                
            fpwrite.write("epoch:{}, lr: {:.6f}, loss: {:.6f}, iters: {}, precision: {:.6f}\n".format(i, lr, loss, j, precision))
            fpwrite.flush()
            for l in range(len(layers)-1, -1, -1):
                delta = layers[l].backward(delta, lr)
        acc = 0
        length = 0
        k = np.arange(len(testdata))
        np.random.shuffle(k)
        testdata = testdata[k]
        test_l = test_l[k]
        testlabel = testlabel[k]
        if i==epoch-1:
            num = len(testdata)
        else:
            num = len(testdata)//(1000)
        dic = {i:0 for i in range(10)}
        for j in range(num):
            images = testdata[j*batchsize:(j+1)*batchsize, :, :]
            images = images[:, np.newaxis, :, :]
            label = testlabel[j*batchsize:(j+1)*batchsize, :]
            label_single = test_l[j*batchsize:(j+1)*batchsize]
            for l in range(len(layers)):
                images = layers[l].forward(images)
            loss, delta, predict = cross_entropy_loss(images, label)
            p = np.argmax(predict, axis=-1)
            length += len(label_single)
            acc += np.sum(label_single==p)
            
            for ij in range(len(p)):
                if p[ij]==label_single[ij]:
                    dic[p[ij]] += 1
            
        precision = acc / length
        meanloss = meanloss / iters
        # savemodel
        allmodel = []
        for l in layers:
            k = dir(l)
            if 'restore_model' in k and 'save_model' in k:
                allmodel.append(l.save_model())
        name = "epoch_"+str(i)+"_loss_"+str(round(meanloss, 6))+"_pre_"+str(round(precision, 6))+"_%s.pkl"%choose
        
        with open(os.path.join(modelpath, name), 'wb') as obj:
            pickle.dump(allmodel, obj)
            
        dic['precision'] = precision
        df = pd.DataFrame(dic, index=np.arange(1)).T
        df.to_csv(os.path.join(abspath, name.replace(".pkl", ".csv")), index=True)

        fpwrite.write("epoch: {}, testset precision: {}\n\n".format(i, precision))
        fpwrite.flush()

if __name__ =="__main__":
    savepath = abspath
    choose = 'large'           #   [mid, small, small_bn, large, large_bn, morelarge]
    pretrained_model = r'C:\Users\10696\Desktop\access\numpy_cnn\model\epoch_0_loss_0.329452_pre_0.90 8_large.pkl'
    logdir = os.path.join(savepath, 'log')
    logfile = os.path.join(logdir, 'log_alexnet_mnist_%s.txt'%choose)
    fpwrite = open(logfile, 'w', encoding='utf-8')
    
    Alexnet_train(10)
    fpwrite.close()