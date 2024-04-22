import os, sys, glob
import numpy as np
import torch

"""
from: https://github.com/gr-b/W-Net-Pytorch/blob/master/

Each module consists of two 3 x 3 conv layers, each followed by a ReLU non-linearity and batch normalization.

In the expansive path, modules are connected via transposed 2D convolution
layers.

The input of each module in the contracting path is also bypassed to the
output of its corresponding module in the expansive path

we double the number of feature channels at each downsampling step
We halve the number of feature channels at each upsampling step

"""

# options
BatchNorm = True
InstanceNorm = True
Dropout = True
k = 2

class ConvBlock(torch.nn.Module):
    
    def __init__(self,
                 input_dim,
                 output_dim):
        
        # initialize the nn.module class with super().__init__()
        super(ConvBlock, self).__init__()

        # define the layers of the block
        layers = [
            torch.nn.Conv2d(input_dim, output_dim, 1), # 1x1 conv through all channels
            torch.nn.Conv2d(output_dim, output_dim, 3, padding=1, groups=output_dim), # 3x3 conv through each channel
            torch.nn.InstanceNorm2d(output_dim),
            torch.nn.BatchNorm2d(output_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Conv2d(output_dim, output_dim, 1),
            torch.nn.Conv2d(output_dim, output_dim, 3, padding=1, groups=output_dim),
            torch.nn.InstanceNorm2d(output_dim),
            torch.nn.BatchNorm2d(output_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
        ]

        # if instancenorm, batchnorm, or dropout are not true, don't include
        if not InstanceNorm:
            layers = [layer for layer in layers if not isinstance(layer, torch.nn.InstanceNorm2d)]
        if not BatchNorm:
            layers = [layer for layer in layers if not isinstance(layer, torch.nn.BatchNorm2d)]
        if not Dropout:
            layers = [layer for layer in layers if not isinstance(layer, torch.nn.Dropout)]

        # create module from layers
        self.module = torch.nn.Sequential(*layers)

    # forward pass
    def forward(self, x):
        return self.module(x)

class BaseNet(torch.nn.module): # define singular U-Net block

    def __init__(self, input_channels=3,
                 encoder = [64, 128, 256, 512],
                 decoder = [1024, 512, 256],
                 output_channels = k):
        
        # init nn.module class
        super(BaseNet, self).__init__()

        layers = [
            torch.nn.conv2d(input_channels, 64, 3, padding = 1),
            torch.nn.InstanceNorm2d(64),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),

            torch.nn.conv2d(64, 64, 3, padding = 1),
            torch.nn.InstanceNorm2d(64),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
        ]

        # abide by options
        if not InstanceNorm:
            layers = [layer for layer in layers if not isinstance(layer, torch.nn.InstanceNorm2d)]
        if not BatchNorm:
            layers = [layer for layer in layers if not isinstance(layer, torch.nn.BatchNorm2d)]
        if not Dropout:
            layers = [layer for layer in layers if not isinstance(layer, torch.nn.Dropout)]
        
        # create first block
        self.first_module = torch.nn.Sequential(*layers)

        # create encoder 
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.enc_modules = torch.nn.ModuleList(
            [ConvBlock(channels, 2*channels) for channels in encoder]
        )

        # create decoder
        decoder_out_sizes = [int(x/2) for x in decoder]

        self.dec_transpose_layers = torch.nn.ModuleList(
            [torch.nn.ConvTranspose2d(channels, channels, 2, stride=2) for channels in decoder]
        )
        
        self.dec_modules = torch.nn.ModuleList(
            [ConvBlock(3*channels_out, channels_out) for channels_out in decoder_out_sizes]
        )

        self.last_dec_transpose_layer = torch.nn.ConvTranspose2d(128, 128, 2, stride = 2)

        layers = [
            torch.nn.Conv2d(128+64, 64, 3, padding=1),
            torch.nn.InstanceNorm2d(64),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),

            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.InstanceNorm2d(64),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),

            torch.nn.Conv2d(64, output_channels, 1),
            torch.nn.ReLU(),
        ]

        # abide by options
        if not InstanceNorm:
            layers = [layer for layer in layers if not isinstance(layer, torch.nn.InstanceNorm2d)]
        if not BatchNorm:
            layers = [layer for layer in layers if not isinstance(layer, torch.nn.BatchNorm2d)]
        if not Dropout:
            layers = [layer for layer in layers if not isinstance(layer, torch.nn.Dropout)]

        self.last_module = torch.nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.first_module(x)
        activations = [x1]

        for module in self.enc_modules:
            activations.append(module(self.pool(activations[-1])))
        
        x_ = activations.pop(-1)

        for conv, upconv in zip(self.dec_modules, self.dec_transpose_layers):
            skip_connection = activations.pop(-1)
            x_ = conv(
                torch.cat((skip_connection, upconv(x_)), dim = 1)
            )
        segmentations = self.last_module(
            torch.cat((activations[-1], self.last_dec_transpose_layer(x_)), 1)
        )

        return segmentations
    
class WNet(torch.nn.module):

    def __init__(self,
                 encoder_layer_sizes = [64, 128, 256, 512],
                 decoder_layer_sizes = [1024, 512, 256]):
        super(WNet, self).__init__()

        self.U_encoder = BaseNet(input_channels=3,
                                 encoder = encoder_layer_sizes,
                                 decoder = decoder_layer_sizes,
                                 output_channels = k)

        self.softmax = torch.nn.Softmax2d()

        self.U_decoder = BaseNet(input_channels =  k,
                                 encoder = encoder_layer_sizes,
                                 decoder = decoder_layer_sizes,
                                 output_channels = 3)
        
        self.sigmoid = torch.nn.Sigmoid()

    def forward_encoder(self, x):
        x9 = self.U_encoder(x)
        segmentations = self.softmax(x9)

        return segmentations

    def forward_decoder(self, x):
        x18 = self.U_decoder(segmentations)
        segmentations = self.sigmoid(x18)
        return segmentations
    
    def forward(self, x): # x is 3 channels, 224x224
        segmentations = self.forward_encoder(x)
        x_prime = self.forward_decoder(segmentations)
        return segmentations, x_prime
        