"""
Module: enhancement
Applies Frangi vesselness filtering and morphological processing to enhance tubular structures.
"""

import matplotlib.pyplot as plt
from skimage.filters import frangi
from skimage.morphology import binary_closing, disk


def frangi_vessel_enhancement(image, visualize=False, sigmas=(1, 2, 3), threshold=0.2, closing_radius=5):
    """
    Enhance vessel-like structures in the input image using the Frangi filter.

    Args:
        image (ndarray): Grayscale input image.
        visualize (bool): If True, shows intermediate processing results.
        sigmas (tuple): Sigma values for multi-scale Frangi filter.
        threshold (float): Threshold to binarize vesselness response.
        closing_radius (int): Radius for morphological closing to connect vessels.

    Returns:
        ndarray: Binary mask of enhanced vessel regions.
    """
    vesselness = frangi(image, sigmas=sigmas)
    binary = vesselness > threshold
    closed = binary_closing(binary, disk(closing_radius))

    if visualize:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Original")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(vesselness, cmap='hot')
        plt.title("Frangi Vesselness")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(closed, cmap='gray')
        plt.title("Threshold + Closing")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return closed
