# import packages 
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image, ImageDraw, ImageFile
from tqdm.auto import tqdm
from statistics import stdev



def preprocess_images(image_names,
                     image_path,
                     save_path,
                     verbose=True,
                     save=True,
                     plot=True):
  
  # instantiate crop points

  


  
  for image_name, cut_pt in zip(tqdm(image_names), y_cut_pts):
    if verbose:
      print(f'Processing image: {image_name}')
      
    # Load the image
    image = cv2.imread(image_path + image_name)

    # Define the desired ranges for x and y axes
    x_start, x_end = 0, image.shape[1]
    y_start, y_end = cut_pt, image.shape[0]

    # Perform cropping
    cropped_image = image[y_start:y_end, x_start:x_end]

    if save:
      # Save the cropped image
      cv2.imwrite(save_path + image_name, cropped_image)

    if plot:
      fig, ax = plt.subplots(1, 2, figsize=(10, 5))
      ax[0].imshow(image)
      ax[0].set_title('Original Image')
      ax[1].imshow(cropped_image)
      ax[1].set_title('Cropped Image')
      plt.show()

  return None
