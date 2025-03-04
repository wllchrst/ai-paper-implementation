# Residual Attention Network Architecture

**Architecture's paper link: https://arxiv.org/pdf/1704.06904 .**

## Related Works

- **Attention Mechanisms**, shows evidence from human perception highlights the importance of attention mechanisms, that help the process of top down information to guid bottom up feedforward process.
- **RNN** and **LSTM** uses attention to gather top information to decide where to attend the next feature learning steps.

## Architecture

### Attention Module

- Each attention module layer of this network consists of two different branches which is **trunk** and **mask** branch which serve two different purpose. Trunk branch is used for **feature processing** where in the paper just a number of residual unit layer. Mask branch is the complicated part where it is used as **control gates** like in (LSTM) for neurons of the layer trunk branch.
- **Attention Residual Learning**, naive stacking of attention module can lead to lost or degrade of important features in deep layers, because of the dot product with value rangin from 0 - 1. The author proposes a new function to combine the two branch from an attention module mentioned in the paper.
- **Soft Mask Branch**, the mask that is going to be used for this network, which includes several max pooling operation to the feature set, to increase the receptive field rapidly. Linear interpolation are also done after some residual unit layer. Then a sigmoid layer normalizes the output after two consecutive 1x1 Convolution Layer.
- **Spatial Attention** and **Channel Attention** are both constraint that can be added to the soft mask branch by changing the normalization step before the activation function. Channel Attention focused on doing normalization with all channel in the context, but spatial attention is done by doing normalization each channel at a time. Author did experiment with three different types of attention normalization using spatial, channel and also mixed attention which all formulas of each function can be found in the paper.
