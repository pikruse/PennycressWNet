import torch
import torch.nn as nn
import torch.nn.functional as F

class Sobel(torch.nn.Module):
    
    def __init__(self, channels = 4):
        
        # initialize
        super().__init__()
        self.channels = channels
        
        # sobel operators
        self.G_x = torch.tensor(
            [[[[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]]]).repeat(1, self.channels, 1, 1).float()
        self.G_y = torch.tensor(
            [[[[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]]]]).repeat(1, self.channels, 1, 1).float()
            
    def to(self, device):
        self.G_x = self.G_x.to(device)
        self.G_y = self.G_y.to(device)
        return self
        
    def forward(self, input):

        G_x = torch.empty_like(input)
        G_y = torch.empty_like(input)
        
        # compute sobel convolutions
        if self.G_x.shape[1] == input.shape[1]:
            G_x = F.conv2d(input, self.G_x)
            G_y = F.conv2d(input, self.G_y)
        else:
            for i in range(input.shape[1]):
                G_x[:,i:i+1] = F.conv2d(input[:,i:i+1], self.G_x)
                G_y[:,i:i+1] = F.conv2d(input[:,i:i+1], self.G_y)
        
        # combine to form approximate gradient magnitude
        G = (G_x**2 + G_y**2)#.sqrt()
        
        return G