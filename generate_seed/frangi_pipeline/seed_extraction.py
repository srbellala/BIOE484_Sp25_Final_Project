"""
Module: seed_extraction
Extracts symmetric seed points from labeled binary masks.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops


def extract_symmetric_seeds_from_region(mask, visualize=False, center_radius_ratio=0.2, vertical_focus='lower'):
    """
    Extract symmetric seed points from segmented binary regions.

    Args:
        mask (ndarray): Binary mask of vessel region.
        visualize (bool): Whether to show extracted seed points.
        center_radius_ratio (float): X-axis range around the image center to restrict region search.
        vertical_focus (str): Which vertical half to focus on ('lower' or 'upper').

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: Coordinates of (seed1, seed2).
    """
    h, w = mask.shape
    labeled = label(mask)
    regions = regionprops(labeled)
    cx = w // 2
    radius_x = int(w * center_radius_ratio)

    def is_in_target_region(region, side='left'):
        y0, x0 = region.centroid
        if vertical_focus == 'lower' and y0 <= h // 2:
            return False
        if side == 'left':
            return x0 < cx and abs(x0 - cx) < radius_x
        else:
            return x0 > cx and abs(x0 - cx) < radius_x

    left_regions = [r for r in regions if is_in_target_region(r, side='left')]
    if not left_regions:
        raise RuntimeError("No region found on left side.")
    region1 = max(left_regions, key=lambda r: r.area)
    dist1 = distance_transform_edt(labeled == region1.label)
    seed1 = np.unravel_index(np.argmax(dist1), dist1.shape)

    right_regions = [r for r in regions if is_in_target_region(r, side='right')]
    if not right_regions:
        raise RuntimeError("No symmetric region found on right side.")
    region2 = max(right_regions, key=lambda r: r.area)
    dist2 = distance_transform_edt(labeled == region2.label)
    seed2 = np.unravel_index(np.argmax(dist2), dist2.shape)

    if visualize:
        plt.imshow(mask, cmap='gray')
        plt.plot(seed1[1], seed1[0], 'ro', label='Seed 1')
        plt.plot(seed2[1], seed2[0], 'bo', label='Seed 2')
        plt.title("Symmetric Seed Points")
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return seed1, seed2
