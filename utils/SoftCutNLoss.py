# import packages
import time, os, sys, glob
import numpy as np
import torch
import torch.nn.functional as F

from torch.autograd import Function

os.path.append('../')

# custom imports
from utils.GetLowestGPU import GetLowestGPU``
device = torch.device(GetLowestGPU(verbose=0))

"""
code from: https://github.com/gr-b/W-Net-Pytorch/blob/master/soft_n_cut_loss.py

The weight matrix w is a measure of the weight between each pixel and
every other pixel. so w[u][v] is a measure of
  (a) Distance between the brightness of the two pixels.
  (b) Distance in positon between the two pixels

The NCut loss metric is then:
  (a) association:    sum(weight_connection) * P(both pixels in the connection are in the class)
  (b) disassociation: sum(weight_connection) * P(first pixel is in the class)
  N Cut loss = disassociation / association
"""

def soft_cut_n_loss(inputs, segmentations,
                    k, input_size):

    """
    Calculate the soft N-cut loss for a batch of images

    Parameters:
        inputs (tensor): (batch_size x 3 x h x w) tensor of input images
        segmentations (tensor): (batch_size x h x w x k) tensor of the probability of each pixel being in each class
    
    Returns:
        loss (tensor): single loss value for the batch
    """

    # don't do n-cut loss batch-wise -> instead, split it up and do it instance-wise
    loss = 0

    for i in range(inputs.shape[0]):
        # flatten image
        flatten_image = torch.mean(inputs[i], dim=0)
        flatten_image = flatten_image.reshape(flatten_image.shape[0]**2)
        loss += soft_cut_n_loss_(flatten_image, segmentations[i], k, input_size, input_size)
    
    loss = loss / inputs.shape[0]

    return loss

def soft_cut_n_loss_(flatten_image, prob, k, rows, cols):
    
    """
    Calculate the soft N-cut loss for a single image

    Parameters:
        flatten_image (tensor): 1D tensor of the row-flattened image (intensity is the mean of three channels)
        prob (tensor): (rows x cols x k) tensor of the probability of each pixel being in each class
        k (int): number of classes
        rows (int): number of rows in the original image
        cols (int): number of columns in the original image
    
    Returns:
        soft_n_cut_loss (tensor): loss for single image
    """
    soft_n_cut_loss = k
    weights = edge_weights(flatten_image, rows, cols)

    for t in range(k):
        soft_n_cut_loss = soft_n_cut_loss - (numerator(prob[t, :, ], weights)/denominator(prob[t, :, :], weights))

    return soft_cut_n_loss

def edge_weights(flatten_image, rows, cols,
                 std_intensity = 3, std_position = 1, radius = 5):
    
    """
    Computes 2D tensor of edge weights in the pixel graph

    Parameters:
        flatten_image (tensor): 1D tensor of the row flattened image (intensity is the mean of three channels)
        rows (int): rows in the original image
        cols (int): columns in the original image
        
        std_intensity (int): standard deviation for intensity
        std_position (int): standard deviation for position
        radius (int): distance around each pixel where weights are non-zero
    
    Returns:
        weights (tensor): 2D tensor of edge weights in the pixel graph
    
    Other params:
        n (int): number of pixels in the image
    """

    ones = torch.ones_like (flatten_image, dtype = torch.float32)
    ones.to(device)

    A = outer_product(flatten_image, ones)
    A_T = torch.t(A)
    d = torch.div((A - A_T), std_intensity)
    intensity_weight = torch.exp(-1*torch.mul(d, d))

    xx, yy, torch.meshgrid(torch.arange(rows, dtype=torch.float),
                           torch.arange(cols, dtype=torch.float))

    xx = xx.reshape(rows*cols)
    yy = yy.reshape(rows*cols)

    xx, yy = xx.to(device), yy.to(device)

    ones_xx = torch.ones_like(xx, dtype=torch.float)
    ones_yy = torch.ones_like(yy, dtype = torch.float)

    ones_xx, ones_yy = ones_xx.to(device), ones_yy.to(device)

    A_x = outer_product(xx, ones_xx)
    A_y = outer_product(yy, ones_yy)

    xi_xj = A_x - torch.t(A_x)
    yi_yj = A_y - torch.t(A_y)

    sq_distance_matrix = torch.mul(xi_xj, xi_xj) + torch.mul(yi_yj, yi_yj)

    dist_weight = torch.exp(-torch.div(sq_distance_matrix, std_position**2))
    weight = torch.mul(intensity_weight, dist_weight)

    return weight

def outer_product(v1, v2):

    """
    Computes outer product of two vectors

    Parameters:
        v1 (tensor): m x 1 tensor
        v2 (tensor): m x 1 tensor

    Returns:
        v1 x v2 (tensor): m x m tensor
    """

    v1 = v1.reshape(-1)
    v2 = v2.reshape(-1)

    v1 = torch.unsqueeze(v1, dim=0)
    v2 = torch.unsqueeze(v2, dim=0)

    return (torch.matmul(torch.t(v1), v2))

def numerator(k_class_prob, weights):

    """
    Creates numerator for the soft N-cut loss

    Parameters:
        k_class_prob (tensor): (rows x cols) tensor of the probability of each pixel being in class k
        weights (tensor): 2D tensor of edge weights in the pixel graph

    Returns:
        numerator (tensor): numerator for the soft N-cut loss
    """

    k_class_prob = k_class_prob.reshape(-1)
    a = torch.mul(weights, outer_product(k_class_prob, k_class_prob))

    return torch.sum(a)

def denominator(k_class_prob, weights):

    """
    Computes denominator for the soft N-cut loss

    Parameters:
        k_class_prob (tensor): (rows x cols) tensor of the probability of each pixel being in class k
        weights (tensor): (n x n) 2D tensor of edge weights in the pixel graph 

    Returns:
        denominator for soft N-cut loss
    """

    k_class_prob = k_class_prob.view(-1)
    
    denominator = torch.sum(
        torch.mul(
            weights, outer_product(
                k_class_prob,
                torch.ones_like(k_class_prob)
            )))

    return denominator

def reconstruction_loss(x, x_prime):
    bce = F.binary_cross_entropy(x_prime, x, reduction = 'sum')
    return bce
    