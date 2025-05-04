"""
Shared utility module for loading and converting MRI images to grayscale float format.
"""

import numpy as np
from skimage.io import imread
from skimage.color import rgba2rgb, rgb2gray
from skimage import img_as_float


def load_mri_image(path):
    """
    Load and convert an MRI image to 2D grayscale float format.

    Args:
        path (str): Path to the image file.

    Returns:
        ndarray: Preprocessed grayscale image as float.
    """
    img = imread(path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = rgba2rgb(img)
    if img.ndim == 3:
        img = rgb2gray(img)
    return img_as_float(img)
