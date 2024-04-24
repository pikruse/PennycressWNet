# import packages 
import seaborn as sns
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image, ImageDraw, ImageFile
from statistics import stdev



def preprocess_image(image_names,
                     image_path,
                     save_path,
                     verbose=True,
                     save=True,
                     plot=True):
  
  # instantiate crop points
  y_cut_pts=[4650, 2400, 5000, 9825, 1090, 2150, 2500, 2850]
  

  if verbose:
    print(f'Processing image: {image_name}')
  
  for image_name, cut_pt in zip(image_names, y_cut_pts):
    # Load the image
    image = Image.open(image_path + image_name)

    if plot:
      # Display the image
      plt.imshow(image)
      plt.show()

    # Construct the full path to the image
    image_path = os.path.join(leaf_folder, file)
    image = cv2.imread(image_path)

    # Define the desired ranges for x and y axes
    x_start, x_end = 0, image.shape[1]
    y_start, y_end = cut_pt, image.shape[0]

    # Perform cropping
    cropped_image = image[y_start:y_end, :]

    if save:
      # Save the cropped image
      Image.save(save_path + image_name, cropped_image)

    if plot:``
      fig, ax = plt.subplots(1, 2, figsize=(10, 5))
      ax[0].imshow(image)
      ax[0].set_title('Original Image')
      ax[1].imshow(cropped_image)
      ax[1].set_title('Cropped Image')
      plt.show()

  return None
