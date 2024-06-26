# import packages
import time, os, sys, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
from scipy.ndimage import grey_opening
from importlib import reload

sys.path.append('../')

# custom imports
from utils.GetLowestGPU import GetLowestGPU
import utils.Filter as Filter

reload(Filter)

class NCutLoss2D(nn.Module):
    r"""Implementation of the continuous N-Cut loss, as in:
    'W-Net: A Deep Model for Fully Unsupervised Image Segmentation', by Xia, Kulis (2017)"""

    def __init__(self, device, radius: int = 4, sigma_1: float = 5, sigma_2: float = 1):
        r"""
        :param radius: Radius of the spatial interaction term
        :param sigma_1: Standard deviation of the spatial Gaussian interaction
        :param sigma_2: Standard deviation of the pixel value Gaussian interaction
        """
        super(NCutLoss2D, self).__init__()
        self.radius = radius
        self.device = device
        self.sigma_1 = sigma_1  # Spatial standard deviation
        self.sigma_2 = sigma_2  # Pixel value standard deviation

    def forward(self, labels: torch.Tensor, inputs:torch.Tensor) -> torch.Tensor:
        r"""Computes the continuous N-Cut loss, given a set of class probabilities (labels) and raw images (inputs).
        Small modifications have been made here for efficiency -- specifically, we compute the pixel-wise weights
        relative to the class-wide average, rather than for every individual pixel.

        :param labels: Predicted class probabilities
        :param inputs: Raw images
        :return: Continuous N-Cut loss
        """
        num_classes = labels.shape[1]
        kernel = Filter.gaussian_kernel(radius=self.radius, sigma=self.sigma_1, device=self.device)
        loss = 0

        for k in range(num_classes):
            # Compute the average pixel value for this class, and the difference from each pixel
            class_probs = labels[:, k].unsqueeze(1)
            class_mean = torch.mean(inputs * class_probs, dim=(2, 3), keepdim=True) / \
                torch.add(torch.mean(class_probs, dim=(2, 3), keepdim=True), 1e-5)
            diff = (inputs - class_mean).pow(2).sum(dim=1).unsqueeze(1)

            # Weight the loss by the difference from the class average.
            weights = torch.exp(diff.pow(2).mul(-1 / self.sigma_2 ** 2))

            # Compute N-cut loss, using the computed weights matrix, and a Gaussian spatial filter
            numerator = torch.sum(class_probs * F.conv2d(class_probs * weights, kernel, padding=self.radius))
            denominator = torch.sum(class_probs * F.conv2d(weights, kernel, padding=self.radius))
            loss += nn.L1Loss()(numerator / torch.add(denominator, 1e-6), torch.zeros_like(numerator))

        return num_classes - loss

    
class OpeningLoss2D(nn.Module):

    """
    from: https://github.com/fkodom/wnet-unsupervised-image-segmentation/tree/master

    Computes the Mean Squared Error between computed class probabilities their grey opening.
    Grey opening is a morphology operation, which performs an erosion followed by dilation.  Conceptually, this encourages the network
    to return sharper boundaries to objects in the class probabilities.
    """

    def __init__(self, device, radius: int = 2):
        """
        Parameters:
            radius (int): radius for channel-wise grey opening operation
            device (torch.device): gpu to run the operation on
        """

        super(OpeningLoss2D, self).__init__()
        self.radius = radius
        self.device = device

    def forward(self, labels:torch.Tensor, *args) -> torch.Tensor:
        """
        Computes opening loss (MSE due to performing a greyscale opening operation)

        Parameters:
            labels (torch.Tensor): class probabilities
            args: additional arguments if user provides input/output image values

        Returns:
            torch.Tensor: the opening loss
        """

        smooth_labels = labels.clone().detach().cpu().numpy()
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                smooth_labels[i, j] = grey_opening(smooth_labels[i, j], self.radius)

        smooth_labels = torch.from_numpy(smooth_labels.astype(np.float32))
        smooth_labels = smooth_labels.to(self.device)

        return nn.MSELoss()(labels, smooth_labels.detach())


def reconstruction_loss(x, x_prime):
    bce = F.binary_cross_entropy(x_prime, x, reduction = 'mean')
    return bce
    