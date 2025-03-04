import torch.nn as nn

class AttentionModule(nn.Module):
    '''
    Attention Module for the Residual Attention Network
    '''
    def __init__(self):
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2)
