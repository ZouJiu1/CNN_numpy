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

from torchvision import datasets
from PIL import Image
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

abspath = os.path.abspath(__file__)
filename = abspath.split(os.sep)[-1]
abspath = abspath.replace(filename, "")

def loading_model(num_classes):
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
    if os.path.exists(pretrained_model):
        with open(pretrained_model, 'rb') as obj:
            models = pickle.load(obj)
        cnt = 0
        for l in layers:
            k = dir(l)
            if 'restore_model' in k and 'save_model' in k:
                l.restore_model(models[cnt])
                cnt += 1
    return layers

def predict_evaluate(layers):
    batchsize = 100
    datapath = os.path.join(abspath, 'dataset')
    os.makedirs(datapath, exist_ok=True)
    
    datatest = datasets.MNIST(root = datapath, train=False, download=True)
    testdata, testlabel = datatest._load_data()
    # */255
    testdata, testlabel = testdata.cpu().numpy() / 255, testlabel.cpu().numpy()
    #one-hot
    test_label = np.zeros((len(testlabel), 10))
    test_label[range(len(testlabel)), testlabel] = 1
    test_l = testlabel.copy()
    testlabel = test_label.copy()

    if predict_or_evaluate:
        cvshow = os.path.join(abspath, 'cvshow')
        os.makedirs(cvshow, exist_ok = True)
        for i in os.listdir(cvshow):
            os.remove(os.path.join(cvshow, i))
        for i in range(len(testlabel)):
            img = testdata[i, :, :]
            ori = (deepcopy(img)[:, :]*255).astype(np.uint8)
            img = img[np.newaxis, np.newaxis, :, :]
            truth = test_l[i]
            for l in range(len(layers)):
                img = layers[l].forward(img)
            p_shift = img - np.max(img, axis = -1)[:, np.newaxis]   # avoid too large in exp 
            predict = np.exp(p_shift) / np.sum(np.exp(p_shift), axis = -1)[:, np.newaxis]
            p = np.argmax(predict, axis=-1)[0]
            plt.imshow(ori)
            plt.title("Predict:"+str(p)+", Truth:"+str(truth))
            plt.savefig(os.path.join(cvshow, str(i)+"_p_"+str(p)+"_t_"+str(truth)+ ".jpg"), bbox_inches='tight')
            image = Image.fromarray(ori).convert("L")
            image.save(os.path.join(cvshow, str(i)+"_Predict_"+str(p)+"_Truth_"+str(truth)+ ".jpg"))
            if i > 10:
                break
    else:
        dic = {i:0 for i in range(10)}
        acc = 0
        length = 0
        for j in range(len(test_l)):
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
        name = "epoch_"+str(i)+"_pre_"+str(round(precision, 6))+"_%s.csv"%choose
        dic['precision'] = precision
        df = pd.DataFrame(dic, index=np.arange(1)).T
        df.to_csv(os.path.join(abspath, name), index=True)

if __name__ =="__main__":
    savepath = abspath
    choose = 'large'               #   mid   small    large     morelarge
    pretrained_model = r'C:\Users\10696\Desktop\access\numpy_cnn\model\epoch_0_loss_0.422414_pre_0.883_large.pkl'
    layers = loading_model(10)
    predict_or_evaluate = False
    predict_evaluate(layers)