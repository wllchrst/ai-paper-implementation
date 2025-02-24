import torch.nn as nn

class AlexNet(nn.Module):
    """
    Class for Alex Net Architecture

    Notes for the alex net architecture.
    - ReLU Layer is applied to every Convolutional and Full Connected Layer
    - Local Response Normalization is used on the first and second Convolutional Layer.
    - Max Pooling Layer always follow Local Response Normalization layer and on the fifth convolutional layer.
    - Paper doesn't specify the stride for every conv layer, only explain the first conv layer stride.
    """
    def __init__(self, input_channel):
        super(AlexNet, self).__init__()
        self.local_response_norm_size = 2

        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=96, kernel_size=11, stride=4),
            nn.Dropout2d(),
            nn.LocalResponseNorm(self.local_response_norm_size, k=2, alpha=10e4, beta=0.75),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.second_layer = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1),
            nn.Dropout2d(),
            nn.LocalResponseNorm(self.local_response_norm_size, k=2, alpha=10e4, beta=0.75),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.third_layer = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fourth_layer = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fifth_layer = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=256, out_features=4096),
            nn.Linear(in_features=4096, out_features=10)
        )

    def forward(self, inputs):
        print(f'Initial shape {inputs.shape}')

        inputs = self.first_layer(inputs)
        print(f'Shape after First Layer: {inputs.shape}')

        inputs = self.second_layer(inputs)
        print(f'Shape after Second Layer: {inputs.shape}')

        inputs = self.third_layer(inputs)
        print(f'Shape after Third Layer: {inputs.shape}')

        inputs = self.fourth_layer(inputs)
        print(f'Shape after Fourth Layer: {inputs.shape}')

        inputs = self.fifth_layer(inputs)
        print(f'Shape after Fifth Layer: {inputs.shape}')

        out = inputs.view(inputs.size(0), -1)

        out = self.fully_connected(out)
        print(f'Shape after Last Layer: {out.shape}')
        return out
