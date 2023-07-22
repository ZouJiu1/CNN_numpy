# CNN in numpy 
## train
I write a cnn network in numpy fully, including forward and backpropagation.<br><br>
including those layers, **convolution**, **AvgPooling**, **MaxPooling**, **Fullconnect**, <br>
**flatten**, **Relu**, **dropout**, **batchnorm**, **Cross Entropy loss** and **MSE loss**<br><br>
In training, it use cpu and slowly, so I write three model with different size, the network is modified by alexnet<br><br>
Training it with MNIST dataset, **it’s precision can > 90%**, it's training now<br><br>
this codes provide functions to save model and restore model to train<br><br>
you can find those models in model dir<br><br>
Train with command<br><br>
```
python train_alexnet_mnist.py
```

### predict

```
python predict.py
```

### precision

| morelarge classes | precision |
| ------ | ------ |
| 0 | 0.9887755102040816 |
| 1 | 0.9876651982378855 |
| 2 | 0.9689922480620154 |
| 3 | 0.9633663366336633 |
| 4 | 0.9592668024439919 |
| 5 | 0.9495515695067265 |
| 6 | 0.9728601252609603 |
| 7 | 0.938715953307393 |
| 8 | 0.9332648870636551 |
| 9 | 0.9534192269573836 |
| all precision | 0.962 |

## blogs
[https://zhuanlan.zhihu.com/p/642481307](https://zhuanlan.zhihu.com/p/642481307)<br>

总共实现了这几个层：

convolution层：[Convolution卷积层的前向传播和反向传播 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/642200457)

AvgPooling层：[AvgPooling平均池化层的前向传播和反向传播 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/642168553)

MaxPooling层：[MaxPooling最大池化层的前向传播和反向传播 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/642116285)

Fullconnect层：[全连接层的前向传播和反向传播 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/642043155)

Cross Entropy和MSE损失函数层：[损失函数的前向传播和反向传播 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/642025009)

flatten层和Relu层：[flatten层和Relu层的前向传播和反向传播 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/642418295)

dropout层：[dropout层的前向传播和反向传播 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/642418780)

层batchnorm：[层batchnorm的前向传播和反向传播 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/642444380)

