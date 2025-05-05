#!/usr/bin/env python3
"""
Uterus Segmentation Pipeline

Loads three DICOM volumes, generates initial contours via multiple edge detectors
on specified slices, refines each with an active contour (snake), and optionally
runs a simple Otsu+snake segmentation through each volume.
"""

import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from skimage import filters, feature, morphology, measure
from skimage.filters import gaussian
from skimage.restoration import denoise_bilateral
from skimage.segmentation import active_contour
from skimage.morphology import remove_small_objects, binary_closing, disk
from skimage.draw import polygon

# --- Helper Functions ---

def load_volume(dicom_path):
    """
    Load a (possibly multi-frame) DICOM into a 3D numpy array: (z, y, x).
    """
    ds = pydicom.dcmread(dicom_path)
    vol = ds.pixel_array
    if vol.ndim == 2:
        vol = vol[np.newaxis, ...]
    return vol

def preprocess_slice(slice_img, sigma=1.0):
    """
    Gaussian smoothing + bilateral denoising to suppress noise/artifacts.
    """
    sm = gaussian(slice_img, sigma=sigma)
    return denoise_bilateral(sm, sigma_color=0.05, sigma_spatial=15, multichannel=False)

def has_uterus(slice_img, min_area=2000):
    """
    Quick Otsu threshold to detect if a slice likely contains the uterus.
    """
    bw = slice_img > filters.threshold_otsu(slice_img)
    bw = remove_small_objects(bw, min_size=min_area)
    return bw.sum() >= min_area

def contour_to_mask(contour, shape):
    """
    Rasterize an (N,2) contour into a binary mask array of given shape.
    """
    rr, cc = polygon(contour[:,0], contour[:,1], shape)
    mask = np.zeros(shape, bool)
    mask[rr, cc] = True
    return mask

def refine_with_snake(slice_img, init_snake,
                      alpha=0.015, beta=10, gamma=0.001, iterations=2500):
    """
    Run the active contour (snake) algorithm to refine an initial contour.
    """
    return active_contour(slice_img, init_snake,
                          alpha=alpha, beta=beta,
                          gamma=gamma, max_iterations=iterations)

def edge_seed_contour(slice_img, method='canny', **kw):
    """
    Generate an initial contour via various edge detectors.
    Returns an (N,2) array of (row, col) points.
    """
    if method == 'sobel':
        edges = filters.sobel(slice_img)
    elif method == 'scharr':
        edges = filters.scharr(slice_img)
    elif method == 'prewitt':
        edges = filters.prewitt(slice_img)
    elif method == 'roberts':
        edges = filters.roberts(slice_img)
    elif method == 'log':
        edges = filters.laplace(gaussian(slice_img, sigma=kw.get('sigma',2)))
    elif method == 'dog':
        g1 = gaussian(slice_img, sigma=kw.get('sigma1',1))
        g2 = gaussian(slice_img, sigma=kw.get('sigma2',3))
        edges = np.abs(g1 - g2)
    elif method == 'canny':
        edges = feature.canny(slice_img,
                              sigma=kw.get('sigma',2),
                              low_threshold=kw.get('low',0.1),
                              high_threshold=kw.get('high',0.3))
    elif method == 'morph_grad':
        se = disk(kw.get('radius',3))
        edges = morphology.dilation(slice_img, se) - morphology.erosion(slice_img, se)
    else:
        raise ValueError(f"Unknown edge method: {method}")

    # Binarize & cleanup
    bw = edges > filters.threshold_otsu(edges)
    bw = remove_small_objects(bw, min_size=kw.get('min_size',500))
    bw = binary_closing(bw, disk(kw.get('close_radius',3)))

    contours = measure.find_contours(bw, 0.5)
    if not contours:
        raise RuntimeError(f"No contour found using {method}")
    # Pick the longest contour as the seed
    return max(contours, key=lambda c: c.shape[0])


# --- Main Pipeline ---

if __name__ == "__main__":
    # 1) Specify the three DICOM series and target slices
    series_paths = {
        'series11': "/data/AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_11_E6_P1_EnIm1.dcm",
        'series3':  "/data/AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_3_E6_P1_EnIm1.dcm",
        'series24': "/data/AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_24_E3_P2_EnIm1.dcm"
    }
    initial_slices = {'series11': 13, 'series3': 14, 'series24': 17}

    # 2) Load volumes
    volumes = {k: load_volume(p) for k, p in series_paths.items()}

    # 3) Define edge methods to try
    methods = ['sobel', 'scharr', 'prewitt', 'roberts', 'log', 'dog', 'canny', 'morph_grad']

    # 4) Generate seeds & refine with snakes on the specified initial slices
    initial_seeds = {}
    refined_snakes = {}

    for key, vol in volumes.items():
        idx = initial_slices[key]
        slice_img = vol[idx]
        proc = preprocess_slice(slice_img)

        initial_seeds[key] = {}
        refined_snakes[key] = {}

        for method in methods:
            try:
                seed = edge_seed_contour(proc, method=method,
                                         sigma=2, low=0.1, high=0.3,
                                         min_size=500, close_radius=3)
                snake = refine_with_snake(proc, seed,
                                          alpha=0.015, beta=10,
                                          gamma=0.001, iterations=2500)

                initial_seeds[key][method] = seed
                refined_snakes[key][method] = snake

            except Exception as e:
                print(f"[{key} slice {idx}] {method} failed: {e}")

    # 5) (Optional) Run a simple Otsu-seeding + snake through each volume
    full_masks = {}
    for key, vol in volumes.items():
        masks = [None] * vol.shape[0]
        for i, sl in enumerate(vol):
            if not has_uterus(sl, min_area=2000):
                continue
            proc = preprocess_slice(sl)

            # Otsu seed
            try:
                t = filters.threshold_otsu(proc)
                bw = proc > t
                bw = binary_closing(bw, disk(3))
                bw = remove_small_objects(bw, min_size=1000)
                contour = max(measure.find_contours(bw, 0.5), key=len)
                snake = refine_with_snake(proc, contour,
                                          alpha=0.015, beta=10,
                                          gamma=0.001, iterations=1000)
                masks[i] = contour_to_mask(snake, proc.shape)
            except:
                continue

        full_masks[key] = masks

    # 6) Visualize seeds vs. refined snakes for one series
    key = 'series11'
    idx = initial_slices[key]
    proc = preprocess_slice(volumes[key][idx])

    fig, axes = plt.subplots(2, len(methods), figsize=(4*len(methods), 8))
    for j, method in enumerate(methods):
        axes[0,j].imshow(proc, cmap='gray')
        axes[0,j].set_title(f"{method} seed")
        axes[0,j].axis('off')

        axes[1,j].imshow(proc, cmap='gray')
        if method in initial_seeds[key]:
            axes[1,j].plot(initial_seeds[key][method][:,1],
                           initial_seeds[key][method][:,0], '--r')
            axes[1,j].plot(refined_snakes[key][method][:,1],
                           refined_snakes[key][method][:,0], '-b')
        axes[1,j].set_title(f"{method} refined")
        axes[1,j].axis('off')

    plt.tight_layout()
    plt.show()
