"""
Uterus Segmentation Pipeline with propagated full-volume segmentation,
per-method plotting saved to disk, terminal checkpoints, and final visualizations.

1) Loads three DICOM volumes.
2) Generates initial contours (seed + snake) via multiple edge detectors on specified slices,
   saving a plot per method.
3) Runs Otsu-based initial contour on each "seed" slice, then propagates that snake
   ±5 slices by using the previous slice's refined snake as the initialization for the next.
4) Saves a final overlay visualization for each series showing the propagated segmentations.
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
output_dir = "output_plots_tst2"
final_output_dir = "output_final_tst2"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(final_output_dir, exist_ok=True)

# ROI criteria for seed location
ROW_FRAC_LOW = 0.6    # lowest contour point must be below 60% height
COL_FRAC_WIDTH = 0.25 # within center ±25% width

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
    tol = W*COL_FRAC_WIDTH
    valid = []
    for c in contours:
        r_max = c[:,0].max()
        col_at_max = c[c[:,0].argmax(),1]
        if r_max>=H*ROW_FRAC_LOW and abs(col_at_max-center)<=tol:
            valid.append(c)
    return valid or contours

def edge_seed(img, method, **kw):
    if method=='sobel': e=filters.sobel(img)
    elif method=='scharr': e=filters.scharr(img)
    elif method=='prewitt': e=filters.prewitt(img)
    elif method=='roberts': e=filters.roberts(img)
    elif method=='log': e=filters.laplace(gaussian(img,sigma=kw.get('sigma',2)))
    elif method=='dog':
        g1=gaussian(img, sigma=kw.get('sigma1',1))
        g2=gaussian(img, sigma=kw.get('sigma2',3))
        e=np.abs(g1-g2)
    elif method=='canny':
        e=feature.canny(img, sigma=kw.get('sigma',2),
                        low_threshold=kw.get('low',0.1),
                        high_threshold=kw.get('high',0.3))
    elif method=='morph_grad':
        se=disk(kw.get('radius',3))
        e=morphology.dilation(img,se)-morphology.erosion(img,se)
    else:
        raise ValueError(f"Unknown method {method}")
    bw = e>filters.threshold_otsu(e)
    bw = remove_small_objects(bw, min_size=kw.get('min_size',500))
    bw = binary_closing(bw, disk(kw.get('close_radius',3)))
    cnts = measure.find_contours(bw,0.5)
    if not cnts:
        raise RuntimeError(f"No contour for {method}")
    cnts = filter_contours(cnts, img.shape)
    return max(cnts, key=lambda c: c.shape[0])

def refine(img, init, alpha=0.015,beta=10,gamma=0.001):
    return active_contour(img, init, alpha=alpha, beta=beta, gamma=gamma)

def contour_to_mask(c, shape):
    rr,cc = polygon(c[:,0],c[:,1],shape)
    m = np.zeros(shape, bool)
    m[rr,cc] = True
    return m

# --- Main ---

if __name__=="__main__":
    # Define series and seed slices
    series = {
        'series11': "data/AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_11_E6_P1_EnIm1.dcm",
        'series3':  "data/AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_3_E6_P1_EnIm1.dcm",
        'series24': "data/AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_1_AWJ24CZBIOHUB_24_E3_P2_EnIm1.dcm"
    }
    seedslice = {'series11':13, 'series3':14, 'series24':17}

    # 1) Load volumes
    print(">> Loading volumes...")
    vols = {k: load_volume(p) for k,p in series.items()}
    print("   Loaded:", ", ".join(vols.keys()))

    # 2) Initial seed+snake per method
    methods = ['sobel','scharr','prewitt','roberts','log','dog','canny','morph_grad']
    init_snakes = {}

    for key, vol in vols.items():
        idx = seedslice[key]
        img = vol[idx]
        p = preprocess(img)
        print(f"\n>> {key}, slice {idx}")

        init_snakes[key] = {}
        for m in methods:
            print(f"  • {m}", end="", flush=True)
            try:
                sd = edge_seed(p, m,
                               sigma=2, low=0.1, high=0.3,
                               min_size=500, close_radius=3)
                sn = refine(p, sd)
                init_snakes[key][m] = sn
                # save plot
                fig, ax = plt.subplots(figsize=(5,5))
                ax.imshow(p, cmap='gray')
                ax.plot(sd[:,1], sd[:,0], '--r', label='Seed')
                ax.plot(sn[:,1], sn[:,0], '-b', label='Snake')
                ax.axis('off'); ax.legend()
                fname = f"{key}_slice{idx}_{m}.png"
                fig.savefig(os.path.join(output_dir,fname),bbox_inches='tight')
                plt.close(fig)
                print(".", end="", flush=True)
            except Exception as e:
                print("x", end="", flush=True)
        print(" done")

    # 3) Propagate Otsu+snake ±5 slices
    print("\n>> Propagating Otsu+snake ±5 slices")
    fullmasks = {}
    for key, vol in vols.items():
        idx = seedslice[key]
        start, end = max(0,idx-5), min(len(vol)-1, idx+5)
        print(f"  • {key}: slices {start}-{end}")

        # initial Otsu snake at idx
        p0 = preprocess(vol[idx])
        t0 = filters.threshold_otsu(p0)
        bw0 = binary_closing(p0>t0, disk(3))
        bw0 = remove_small_objects(bw0, min_size=1000)
        cnt0 = max(measure.find_contours(bw0,0.5), key=len)
        sn0 = refine(p0, cnt0)

        masks = [None]*len(vol)
        masks[idx] = contour_to_mask(sn0, p0.shape)

        # backward
        prev = sn0
        for i in range(idx-1, start-1, -1):
            pi = preprocess(vol[i])
            print(f"    <{i}", end="", flush=True)
            try:
                si = refine(pi, prev)
                masks[i] = contour_to_mask(si, pi.shape)
                prev = si
                print(".", end="", flush=True)
            except:
                print("x", end="", flush=True)

        # forward
        prev = sn0
        for i in range(idx+1, end+1):
            pi = preprocess(vol[i])
            print(f"    >{i}", end="", flush=True)
            try:
                si = refine(pi, prev)
                masks[i] = contour_to_mask(si, pi.shape)
                prev = si
                print(".", end="", flush=True)
            except:
                print("x", end="", flush=True)

        fullmasks[key] = masks
        print()

    # 4) Final visualizations for each series
    print("\n>> Saving final overlay visualizations")
    for key, vol in vols.items():
        masks = fullmasks[key]
        start, end = max(0,seedslice[key]-5), min(len(vol)-1,seedslice[key]+5)
        n = end-start+1
        cols = n
        fig, axes = plt.subplots(1, cols, figsize=(3*cols,3))
        for idx, i in enumerate(range(start, end+1)):
            ax = axes[idx]
            img = vol[i]
            ax.imshow(img, cmap='gray')
            if masks[i] is not None:
                ax.imshow(masks[i], cmap='Reds', alpha=0.3)
            ax.set_title(f"Slice {i}")
            ax.axis('off')
        out_fn = os.path.join(final_output_dir, f"{key}_propagated.png")
        fig.savefig(out_fn, bbox_inches='tight')
        plt.close(fig)
        print(f"  • {key} saved to {out_fn}")

    print("\n>> Pipeline complete.")
