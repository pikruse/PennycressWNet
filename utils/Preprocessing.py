# import necessary packages
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from ipywidgets import FloatProgress
from PIL import Image
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, binary_dilation


def split_image(image_name, 
                image_path = "../data/train/images/", 
                save_path = "../data/train/images_by_pod/",
                plot=False, 
                save=True):
    
    # load image and mask
    image = Image.open(image_path + image_name).convert('RGB')

    # convert to numpy array and normalize
    image = np.array(image) / 255.0

    mask = image.sum(-1) > 1
    mask = binary_dilation(mask, iterations = 2)
    mask = ~mask
    mask = ndimage.binary_fill_holes(mask > 0.5)

    # remove small artifacts
    labels = ndimage.label(mask)[0]
    size = 500
    sizes = np.bincount(labels.reshape(-1))
    for j in range(1, len(sizes)):
        if sizes[j] < size:
            mask[labels == j] = False
    
    # make images fully white bg
    image[~mask] = 1

    # pad image and and mask
    image = np.pad(image, ((100, 100), (100, 100), (0, 0)), mode='edge')
    mask = np.pad(mask, ((100, 100), (100, 100)), mode='constant')

    # label each component in mask and create bounding boxes
    labels = ndimage.label(mask)[0]
    bboxes = ndimage.find_objects(labels)

    if plot:
        # plot image and mask for sanity
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image)
        ax[1].imshow(mask, cmap='gray')
        plt.show()

    # add padding to bounding boxes
    x_pad, y_pad = 100, 100
    for i in range(len(bboxes)):
        x, y = bboxes[i]
        bboxes[i] = slice(x.start-x_pad, x.stop+x_pad), slice(y.start-y_pad, y.stop+y_pad)

    if plot:
        # plot image and mask with bounding boxes
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image)
        axs[1].imshow(mask)
        for bbox in bboxes:
            y, x = bbox
            axs[0].plot([x.start, x.start, x.stop, x.stop, x.start], [y.start, y.stop, y.stop, y.start, y.start], '--', color='r')
            axs[1].plot([x.start, x.start, x.stop, x.stop, x.start], [y.start, y.stop, y.stop, y.start, y.start], '--', color='r')
        plt.tight_layout()
        plt.show()

    # save split images
    for i, bbox in enumerate(tqdm(bboxes)):
        y, x = bbox
        split_image = image[y, x, :]

        # convert to PIL image
        split_image = Image.fromarray((split_image * 255).astype(np.uint8))

        if save:
            # save img, msk
            split_image.save(save_path + image_name[:-4] + "_" + str(i) + ".png")

    return None