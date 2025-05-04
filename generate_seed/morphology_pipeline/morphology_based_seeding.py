'''
Project: Uterus Segmentation Seed Generation
Module: morphology_based_seeding.py

This script generates seed points from a mouse MRI image for uterus segmentation using 
morphological processing and shape heuristics. This version is particularly suitable when 
the abdominal cavity is signal-free.
'''

import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure
from skimage.morphology import opening, remove_small_objects, disk
from skimage.draw import disk as draw_disk
from utils.image_loader import load_mri_image

def generate_seeds(image, area_thresh=500, aspect_ratio_range=(0.3, 3.0), visualize=False):
    """
    Generate seed points for uterus segmentation using thresholding and shape filtering.

    Args:
        image (ndarray): Grayscale MRI image.
        area_thresh (int): Minimum area of regions to consider.
        aspect_ratio_range (tuple): (min_ratio, max_ratio) for region filtering.
        visualize (bool): Whether to visualize intermediate results.

    Returns:
        ndarray: Binary mask containing seed points.
    """
    # Binarize image using Otsu threshold
    binary = image > filters.threshold_otsu(image)

    # Morphological opening to remove small artifacts
    cleaned = opening(binary, disk(3))

    # Remove small objects
    cleaned = remove_small_objects(cleaned, min_size=area_thresh)

    # Label connected components
    label_img = measure.label(cleaned)
    regions = measure.regionprops(label_img)

    # Initialize seed mask
    seeds = np.zeros_like(image, dtype=bool)

    for region in regions:
        if region.area < area_thresh:
            continue
        if region.minor_axis_length == 0:
            continue

        # Compute aspect ratio of the region
        aspect_ratio = region.major_axis_length / region.minor_axis_length
        if not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
            continue

        # Place a small disk at the region centroid as the seed
        cy, cx = map(int, region.centroid)
        rr, cc = draw_disk((cy, cx), radius=2, shape=image.shape)
        seeds[rr, cc] = True

    if visualize:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Original Image')
        ax[1].imshow(cleaned, cmap='gray')
        ax[1].set_title('Morphologically Cleaned')
        ax[2].imshow(image, cmap='gray')
        ax[2].imshow(seeds, cmap='Reds', alpha=0.6)
        ax[2].set_title('Generated Seeds')
        for a in ax:
            a.axis('off')
        plt.tight_layout()
        plt.show()

    return seeds


if __name__ == "__main__":
    # Example usage on a test image (adjust path as needed)
    img_path = "data/3.png"
    image = load_mri_image(img_path)
    seeds = generate_seeds(image, area_thresh=1000, visualize=True)
