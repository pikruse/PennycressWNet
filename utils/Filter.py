import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm

def gaussian_kernel(device: torch.device, radius: int = 3, sigma: float = 4):

    """
    from: https://github.com/fkodom/wnet-unsupervised-image-segmentation/tree/master

    Creates gaussian kernel used in N-cut loss

    Parameters:
        device (torch.device): device to use
        radius (int): radius of the kernel
        sigma (float): standard deviation of the kernel
    Returns:
        torch.Tensor: the gaussian kernel
    """

    x_2 = np.linspace(-radius, radius, 2*radius + 1) ** 2
    dist = np.sqrt(x_2.reshape(-1, 1) + x_2.reshape(1, -1)) / sigma
    kernel = norm.pdf(dist) / norm.pdf(0)
    kernel = torch.from_numpy(kernel.astype(np.float32))
    kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
    kernel = kernel.to(device)

    return kernel