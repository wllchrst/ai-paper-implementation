# Alex Net Architecture
**Papers Link**:
https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html

> Alex Net is an CNN Architecture that is made by three researcher, where the objective of the architecture is to do image classification task.
> The architecture entered the ILSVRC-2012 Competition where it achieved top-5 test error rate of 15.3%compared to 26.2% achieved by the second-best entry.

### Introduction
- Overfitting, because the size of the neural network is really big it is prone to overfit the author mentioned how to fix this issue in the section 4.
- The final result of the Architecture is 5 Convolutional Layer and 3 Full Connected Layer.
- Model is trained with 2 Nvidia GPUS.

### Dataset
Details of the dataset is not going to be explained by detail here.

- Dataset that's used for the model is ImageNet Dataset, consists of variable-resolution image where it is brought down to 256 x 256.

### Architecture
- **ReLU Layer**, in terms of training time other activation layer such as tanh(x) and the sigmoid function is slow because of the saturation that can happen to the result, compared to ReLU Layer where the function is just max(0, x). Also proven a 4 Layer CNN when being trained with CIFAR 10 reach 25% training error first when using ReLU layer.
- Training of the model used to GPU where the model architecture is split into two parts.
- **Local Response Normalization Layer**, is used to generalization through the convolutional layer, and also the layer reduces the author top-1 and top-5 error rates and 1.4% and 1.2% 