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



def preprocess_image(image_name):
  # Load the image
  image = cv2.imread('/content/drive/MyDrive/Leaf_images/leaf2.png')
  # Display the image
  plt.imshow(image)
  plt.show()
  
  
  ImageFile.LOAD_TRUNCATED_IMAGES = True
  
  z=[4650, 2400, 5000, 9825, 1090, 2150, 2500, 2850]
  leaf_images=[]
  
  leaf_folder='/content/drive/MyDrive/Leaf_images/'
  files = os.listdir(leaf_folder)
  leaf_files = [file for file in files if 'leaf' in file]
  # Iterate over the leaf files
  for file, cut in zip(leaf_files, z):
  
    if file == 'test_leaf.png':
      # Construct the full path to the image
      image_path = os.path.join(leaf_folder, file)
      image = mpimg.imread(image_path)
      # Define the desired ranges for x and y axes
      x_start, x_end = 0, image.shape[1]
      y_start, y_end = cut, image.shape[0]
      cropped_image = image[y_start:y_end, x_start:x_end]
      leaf_images.append(cropped_image)
      plt.imshow(cropped_image)
      plt.show()
    else:
      # Construct the full path to the image
      image_path = os.path.join(leaf_folder, file)
      image = cv2.imread(image_path)
      # Define the desired ranges for x and y axes
      x_start, x_end = 0, image.shape[1]
      y_start, y_end = cut, image.shape[0]
      # Perform cropping
      cropped_image = image[y_start:y_end, x_start:x_end]
      leaf_images.append(cropped_image)
      plt.imshow(cropped_image)
      plt.show()
