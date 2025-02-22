import torch.nn as nn

class AlexNet(nn.Module):
    """Class for VVGG 16 Architecture

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_channel):
        super(AlexNet, self).__init__() 
        self.first_conv_layer = nn.Conv2d\
            (in_channels=input_channel, out_channels=96, kernel_size=11, stride=4)

    def forward(self, inputs):
        print(inputs.shape)
        inputs = self.first_conv_layer(inputs)

        print(f"First Layers: {inputs.shape}")

        return inputs
