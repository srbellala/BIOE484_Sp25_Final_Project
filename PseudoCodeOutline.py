# VJ note this is me just putting some blocks of code/fxns that I think could work

import numpy as np 
from skimage import filters, measure
from skimage.filters import gaussian
from skimage.restoration import denoise_bilateral
from skimage.segmentation import active_contour
from skimage.morphology import remove_small_objects, binary_closing, disk




def preprocess_slice(slice_img, sigma=1.0):

    smoothed = gaussian(slice_img, sigma=sigma)
    denoised = denoise_bilateral(smoothed, sigma_color=0.05, sigma_spatial=15, multichannel=False)
    return denoised

def generate_initial_contour(slice_img): #we may want to manually seed this instead, but lets do the naive comp does it all
    #otsu thresholding, cleaning and extract largest contour (we should also pick which direction to start)
    thresh = filters.threshold_otsu(slice_img)
    binary = slice_img > thresh
    binary = binary_closing(binary, disk(3))
    binary = remove_small_objects(binary, min_size=1000)

    contours = measure.find_contours(binary, 0.5)
    if not contours:
        raise RuntimeError("No contours found in mask")
    init_contour = max(contours, key=lambda x: len(x))
    return init_contour
#we can compare this naive approach vs a more manual approach


def refine_with_snake(slice_img, init_snake, alpha = 0.015, beta=10, gamma=0.001, iterations=2500):
    snake = active_contour(
        slice_img,
        init_snake,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        max_iterations=iterations

    )
    return snake


#not all slices will have the uterus so maybe we can get an area or something


#general flow might be:
    # loop over volume
    # filter w/ a initial contour/find uterus fxn
    # preprocess w/ smoothing/denoising
    # generate_initial_contour
    # if fail - fallback to manual_initial_contour
    # refine snake, store contour
    # compare w/ overlap metrics like dice, jaccard (need to have masks for each uterus)


# we'll want to figure out better ways for initial contours - this is probably where the bulk of our time can be spent looking into things, and probably will help with our report
   # sobel, scharr, prewitt, roberts, laplacian of gaussian, difference of gaussian, canny, morph (otsu vs edge seed)


def has_uterus(slice_img, min_area=?): #area based initialization
    return mask.sum() >= min_area


def edge_seed_contour(slice_img, method=?, **kw):
    if method == 'sobel':
           edges = filters.sobel(slice_img)
    elif method == 'scharr':
         edges = filters.scharr(slice_img)
    elif method == 'prewitt':
    elif method == 'roberts':
    elif method == 'log':
    elif method == 'dog':
        g1 = gaussian()
        g2 = gaussian()
        edges = np.abs(g1-g2)
    elif method == 'canny':
        edges = feature.canny()
    elif method == 'morph_grad':
        se = disk(kw.get('radius', 3))
        edges = morphology.dilation(slice_img, se) - morphology.erosion(slice_img, se)
    else: 
        raise ValueError(f"What the heck did you type")
    
    bw = edges > filters.threshold_otsu(edges)
    bw = morphology.remove_small_objects(bw, min_size=kw.get('min_size', 500))

    contours = measure.find_contours(bw, 0.5)

    if not contours:
        raise RuntimeError(f"what the heck no contours from {method}")
    return max(contours, key=len)

def contour_to_mask(contour, shape)
    #rasterize to shape
    return mask