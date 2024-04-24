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



def preprocess_image(image_name,
                     image_path,
                     save_path,
                     verbose=True,
                     save=True,
                     plot=True):
  

  if verbose:
    print(f'Processing image: {image_name}')
  
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
  y_start, y_end = cut, image.shape[0]

  # Perform cropping
  cropped_image = image[y_start:y_end, :]

  if plot:
    plt.imshow(cropped_image)
    plt.show()

  return cropped_image
