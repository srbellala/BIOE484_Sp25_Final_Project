#!/usr/bin/env python3
"""
Uterus Segmentation Pipeline with per-method plotting saved to disk
and terminal checkpoints.

Loads three DICOM volumes, generates initial contours via multiple edge detectors
on specified slices, refines each with an active contour (snake), and saves a plot
after each method. Then runs Otsu+snake segmentation on ±5 slices around each
initial contour. Checkpoint messages are printed to the terminal.
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

# --- Configuration ---
output_dir = "output_plots"
os.makedirs(output_dir, exist_ok=True)

# --- Helper Functions ---

def load_volume(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    vol = ds.pixel_array
    if vol.ndim == 2:
        vol = vol[np.newaxis, ...]
    return vol

def preprocess_slice(slice_img, sigma=1.0):
    sm = gaussian(slice_img, sigma=sigma)
    return denoise_bilateral(sm,
                             sigma_color=0.05,
                             sigma_spatial=15,
                             channel_axis=None)

def has_uterus(slice_img, min_area=2000):
    bw = slice_img > filters.threshold_otsu(slice_img)
    bw = remove_small_objects(bw, min_size=min_area)
    return bw.sum() >= min_area

def contour_to_mask(contour, shape):
    rr, cc = polygon(contour[:,0], contour[:,1], shape)
    mask = np.zeros(shape, bool)
    mask[rr, cc] = True
    return mask

def refine_with_snake(slice_img, init_snake,
                      alpha=0.015, beta=10, gamma=0.001):
    return active_contour(slice_img, init_snake,
                          alpha=alpha, beta=beta, gamma=gamma)

def edge_seed_contour(slice_img, method='canny', **kw):
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

    bw = edges > filters.threshold_otsu(edges)
    bw = remove_small_objects(bw, min_size=kw.get('min_size',500))
    bw = binary_closing(bw, disk(kw.get('close_radius',3)))
    contours = measure.find_contours(bw, 0.5)
    if not contours:
        raise RuntimeError(f"No contour found using {method}")
    return max(contours, key=lambda c: c.shape[0])


# --- Main Pipeline ---

if __name__ == "__main__":
    # 1) Specify DICOMs & seed slices
    series_paths = {
        'series11': "data/AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_11_E6_P1_EnIm1.dcm",
        'series3':  "data/AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_3_E6_P1_EnIm1.dcm",
        'series24': "data/AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_24_E3_P2_EnIm1.dcm"
    }
    initial_slices = {'series11': 13, 'series3': 14, 'series24': 17}

    print(">> Loading volumes...")
    volumes = {k: load_volume(p) for k,p in series_paths.items()}
    print("   Loaded volumes:", ", ".join(series_paths.keys()))

    methods = ['sobel','scharr','prewitt','roberts','log','dog','canny','morph_grad']
    initial_seeds, refined_snakes = {}, {}

    # 2) Process each seed slice
    for key, vol in volumes.items():
        idx = initial_slices[key]
        print(f"\n>> Processing {key}, slice {idx}")
        img = vol[idx]
        proc = preprocess_slice(img)
        print("   - Preprocessed slice")

        initial_seeds[key], refined_snakes[key] = {}, {}
        for method in methods:
            print(f"   - Method '{method}': generating seed...", end="", flush=True)
            try:
                seed = edge_seed_contour(proc,
                                         method=method,
                                         sigma=2, low=0.1, high=0.3,
                                         min_size=500, close_radius=3)
                print("done")
                print("     refining snake...", end="", flush=True)
                snake = refine_with_snake(proc, seed,
                                          alpha=0.015, beta=10, gamma=0.001)
                print("done")

                initial_seeds[key][method] = seed
                refined_snakes[key][method] = snake

                # Save plot instead of showing
                print("     saving plot...", end="", flush=True)
                fig, ax = plt.subplots(figsize=(6,6))
                ax.imshow(proc, cmap='gray')
                ax.plot(seed[:,1], seed[:,0], '--r', label='Seed')
                ax.plot(snake[:,1], snake[:,0], '-b', label='Snake')
                ax.set_title(f"{key} slice {idx} — {method}")
                ax.axis('off')
                ax.legend()
                out_path = os.path.join(output_dir, f"{key}_slice{idx}_{method}.png")
                fig.savefig(out_path, bbox_inches='tight')
                plt.close(fig)
                print(f"saved to {out_path}")

            except Exception as e:
                print(f"FAILED ({e})")

    # 3) Otsu+snake on ±5 slices around each initial contour
    print("\n>> Running Otsu+snake on ±5 slices around initial contour")
    full_masks = {}
    for key, vol in volumes.items():
        idx = initial_slices[key]
        n_slices = vol.shape[0]
        start = max(0, idx - 5)
        end = min(n_slices - 1, idx + 5)
        print(f"   - {key}: slices {start} to {end}")
        masks = [None] * vol.shape[0]
        for i in range(start, end + 1):
            sl = vol[i]
            proc = preprocess_slice(sl)
            print(f"      slice {i}", end="", flush=True)
            if not has_uterus(sl):
                print(" skipped (no uterus)")
                continue
            try:
                t = filters.threshold_otsu(proc)
                bw = proc > t
                bw = binary_closing(bw, disk(3))
                bw = remove_small_objects(bw, min_size=1000)
                contour = max(measure.find_contours(bw,0.5), key=len)
                snake = refine_with_snake(proc, contour,
                                          alpha=0.015, beta=10, gamma=0.001)
                masks[i] = contour_to_mask(snake, proc.shape)
                print(" segmented")
            except Exception as e:
                print(f" failed ({e})")
        full_masks[key] = masks
        print(f"   -> Completed {key}")

    print("\n>> Pipeline finished.")
