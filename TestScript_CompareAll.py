"""
Uterus Segmentation Pipeline

1) Loads three DICOM volumes.
2) For each series and each edge-detection method:
     • Generates an initial contour (seed + snake) on a specified slice
     • Saves a plot of seed vs. refined snake to output_plots/
3) Propagates each method’s snake ±5 slices from the seed slice,
   using the previous slice's snake as initialization.
4) Saves a final overlay montage for each series & method to output_final/.

Terminal checkpoints are printed at each major step.
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
output_plots_dir = "output_plots_TST3"
output_final_dir = "output_final_TST3"
os.makedirs(output_plots_dir, exist_ok=True)
os.makedirs(output_final_dir, exist_ok=True)

# Seed-slice per series
series_paths = {
    'series11': "data/AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_11_E6_P1_EnIm1.dcm",
    'series3':  "data/AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_3_E6_P1_EnIm1.dcm",
    'series24': "data/AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_24_E3_P2_EnIm1.dcm"
}
seed_slices = {'series11':13, 'series3':14, 'series24':17}

# Edge detection methods
methods = ['sobel','scharr','prewitt','roberts','log','dog','canny','morph_grad']

# Location filter parameters
ROW_FRAC_LOW = 0.6
COL_FRAC_WIDTH = 0.25

# --- Helper Functions ---

def load_volume(path):
    ds = pydicom.dcmread(path)
    vol = ds.pixel_array
    return vol[np.newaxis,...] if vol.ndim==2 else vol

def preprocess(img, sigma=1.0):
    sm = gaussian(img, sigma=sigma)
    return denoise_bilateral(sm,
                             sigma_color=0.05,
                             sigma_spatial=15,
                             channel_axis=None)

def filter_contours(contours, shape):
    H, W = shape
    center = W/2
    tol = W * COL_FRAC_WIDTH
    valid = []
    for c in contours:
        r_max = c[:,0].max()
        c_max = c[c[:,0].argmax(),1]
        if r_max >= H*ROW_FRAC_LOW and abs(c_max - center) <= tol:
            valid.append(c)
    return valid if valid else contours

def edge_seed(img, method, **kw):
    if method=='sobel':
        e = filters.sobel(img)
    elif method=='scharr':
        e = filters.scharr(img)
    elif method=='prewitt':
        e = filters.prewitt(img)
    elif method=='roberts':
        e = filters.roberts(img)
    elif method=='log':
        e = filters.laplace(gaussian(img, sigma=kw.get('sigma',2)))
    elif method=='dog':
        g1 = gaussian(img, sigma=kw.get('sigma1',1))
        g2 = gaussian(img, sigma=kw.get('sigma2',3))
        e = np.abs(g1 - g2)
    elif method=='canny':
        e = feature.canny(img,
                          sigma=kw.get('sigma',2),
                          low_threshold=kw.get('low',0.1),
                          high_threshold=kw.get('high',0.3))
    elif method=='morph_grad':
        se = disk(kw.get('radius',3))
        e = morphology.dilation(img, se) - morphology.erosion(img, se)
    else:
        raise ValueError(f"Unknown method '{method}'")
    # Binarize & clean
    bw = e > filters.threshold_otsu(e)
    bw = remove_small_objects(bw, min_size=kw.get('min_size',500))
    bw = binary_closing(bw, disk(kw.get('close_radius',3)))
    contours = measure.find_contours(bw, 0.5)
    if not contours:
        raise RuntimeError(f"No contour found using {method}")
    # Filter by location
    contours = filter_contours(contours, img.shape)
    # Return longest
    return max(contours, key=lambda c: c.shape[0])

def refine(img, snake, alpha=0.015, beta=10, gamma=0.001):
    return active_contour(img, snake, alpha=alpha, beta=beta, gamma=gamma)

def contour_to_mask(contour, shape):
    rr, cc = polygon(contour[:,0], contour[:,1], shape)
    m = np.zeros(shape, bool)
    m[rr, cc] = True
    return m

# --- Main Pipeline ---

if __name__ == "__main__":
    # Load volumes
    print(">> Loading volumes...")
    volumes = {k: load_volume(p) for k,p in series_paths.items()}
    print("   Loaded:", ", ".join(volumes.keys()))

    # Step 1: initial seed+snake per method
    init_snakes = {k:{} for k in volumes}
    for key, vol in volumes.items():
        idx = seed_slices[key]
        print(f"\n>> Processing {key}, slice {idx}")
        proc = preprocess(vol[idx])
        print("   - Preprocessed slice")

        for method in methods:
            print(f"   - Method '{method}': generating seed...", end="", flush=True)
            try:
                seed = edge_seed(proc, method,
                                 sigma=2, low=0.1, high=0.3,
                                 min_size=500, close_radius=3)
                print("done")
                print("     refining snake...", end="", flush=True)
                snake = refine(proc, seed)
                print("done")
                init_snakes[key][method] = snake

                print("     saving plot...", end="", flush=True)
                fig, ax = plt.subplots(figsize=(5,5))
                ax.imshow(proc, cmap='gray')
                ax.plot(seed[:,1], seed[:,0], '--r', label='Seed')
                ax.plot(snake[:,1], snake[:,0], '-b', label='Snake')
                ax.axis('off'); ax.legend()
                out_path = os.path.join(output_plots_dir,
                                        f"{key}_slice{idx}_{method}.png")
                fig.savefig(out_path, bbox_inches='tight')
                plt.close(fig)
                print(f"saved to {out_path}")

            except Exception as e:
                print(f"FAILED ({e})")

    # Step 2: propagate each method’s snake ±5 slices
    propagated = {k:{} for k in volumes}
    for key, vol in volumes.items():
        idx = seed_slices[key]
        start, end = max(0, idx-5), min(len(vol)-1, idx+5)
        print(f"\n>> Propagating {key}: slices {start}–{end}")

        for method, snake0 in init_snakes[key].items():
            print(f"   - Propagating '{method}'", end="", flush=True)
            masks = [None]*len(vol)
            masks[idx] = contour_to_mask(snake0, vol[idx].shape)
            prev = snake0

            # backward
            for i in range(idx-1, start-1, -1):
                print(f" <{i}", end="", flush=True)
                try:
                    proc_i = preprocess(vol[i])
                    sn = refine(proc_i, prev)
                    masks[i] = contour_to_mask(sn, proc_i.shape)
                    prev = sn
                    print("done", end="", flush=True)
                except:
                    print("failed", end="", flush=True)

            # forward
            prev = snake0
            for i in range(idx+1, end+1):
                print(f" >{i}", end="", flush=True)
                try:
                    proc_i = preprocess(vol[i])
                    sn = refine(proc_i, prev)
                    masks[i] = contour_to_mask(sn, proc_i.shape)
                    prev = sn
                    print("done", end="", flush=True)
                except:
                    print("failed", end="", flush=True)

            propagated[key][method] = masks
            print()

    # Step 3: final montage per method
    print("\n>> Saving final montages to", output_final_dir)
    for key, vol in volumes.items():
        idx = seed_slices[key]
        start, end = max(0, idx-5), min(len(vol)-1, idx+5)
        for method, masks in propagated[key].items():
            print(f"   - Creating montage for '{key}' [{method}]", end="", flush=True)
            n = end - start + 1
            fig, axes = plt.subplots(1, n, figsize=(3*n,3), squeeze=False)
            for j, i in enumerate(range(start, end+1)):
                ax = axes[0,j]
                ax.imshow(vol[i], cmap='gray')
                if masks[i] is not None:
                    ax.imshow(masks[i], cmap='Reds', alpha=0.3)
                ax.set_title(f"{i}")
                ax.axis('off')
            out_fn = os.path.join(output_final_dir,
                                  f"{key}_{method}_propagated.png")
            fig.savefig(out_fn, bbox_inches='tight')
            plt.close(fig)
            print(" saved")

    print("\n>> Pipeline complete.")
