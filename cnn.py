import torch
from torch import nn

class VGG(nn.Module):
    def __init__(self, in_channels, num_classes=10, n_blocks=3):

        super(VGG, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # convolutional layers

        self.conv_layers = nn.Sequential(

            # First VGG block
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding='same'),
            nn.ReLU(),  
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding='same'),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2),  

            # Second VGG block
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding='same'),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2),

            # Third VGG block
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # fully connected linear layers
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=4 * 128 * 4 * 4, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(1, -1)
        x = self.linear_layers(x)
        return x

    def init_weights(self):
        torch.nn.init.he_uniform(self.conv_layers.weight)
        self.conv_layers.bias.data.fill_(0.01)
        torch.nn.init.he_uniform(self.linear_layers.weight)
        self.linear_layers.bias.data.fill_(0.01)