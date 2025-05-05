import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgba2rgb, rgb2gray
from skimage import img_as_float, filters, morphology, measure
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, opening, disk
from skimage.draw import disk as draw_disk
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull


def load_mri_image(path):
    img = imread(path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = rgba2rgb(img)
    if img.ndim == 3:
        img = rgb2gray(img)
    return img_as_float(img)


def find_bladder_region(image, intensity_thresh=0.7, area_thresh=1000, visualize=True):
    bright_mask = image > intensity_thresh
    cleaned_mask = remove_small_objects(bright_mask, min_size=area_thresh)
    labeled = label(cleaned_mask)
    props = regionprops(labeled, intensity_image=image)
    best_region = None
    best_score = 0
    for prop in props:
        area = prop.area
        mean_intensity = prop.mean_intensity
        perimeter = prop.perimeter if prop.perimeter > 0 else 1
        circularity = 4 * np.pi * area / (perimeter ** 2)
        score = mean_intensity * circularity * area
        if score > best_score:
            best_score = score
            best_region = prop
    mask = np.zeros_like(image, dtype=bool)
    if best_region:
        coords = best_region.coords
        mask[tuple(coords.T)] = True
        print(f"[Bladder] Area: {best_region.area}, Intensity: {best_region.mean_intensity:.3f}")
    else:
        print("[Bladder] No bladder region found.")
    if visualize:
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap='gray')
        plt.imshow(mask, cmap='Reds', alpha=0.5)
        plt.title("Bladder Detection")
        plt.axis('off')
        plt.show()
    return mask


def generate_seeds(image, area_thresh=500, aspect_ratio_range=(0.3, 3.0), visualize=True):
    binary = image > filters.threshold_otsu(image)
    cleaned = opening(binary, disk(3))
    cleaned = remove_small_objects(cleaned, min_size=area_thresh)
    label_img = measure.label(cleaned)
    regions = measure.regionprops(label_img)
    seeds = np.zeros_like(image, dtype=bool)
    for region in regions:
        if region.area < area_thresh or region.minor_axis_length == 0:
            continue
        aspect_ratio = region.major_axis_length / region.minor_axis_length
        if not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
            continue
        cy, cx = map(int, region.centroid)
        rr, cc = draw_disk((cy, cx), radius=2, shape=image.shape)
        seeds[rr, cc] = True
    if visualize:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Original Image')
        ax[1].imshow(cleaned, cmap='gray')
        ax[1].set_title('Cleaned Binary')
        ax[2].imshow(image, cmap='gray')
        ax[2].imshow(seeds, cmap='Reds', alpha=0.6)
        ax[2].set_title('Generated Seeds')
        for a in ax:
            a.axis('off')
        plt.tight_layout()
        plt.show()
    return seeds


def segment_uterus(image, seeds, visualize=True):
    distance = distance_transform_edt(image)
    marker_labels = measure.label(seeds)
    mask = image > filters.threshold_otsu(image)
    labels = watershed(-distance, markers=marker_labels, mask=mask)
    if visualize:
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap='gray')
        plt.imshow(labels, cmap='jet', alpha=0.4)
        plt.title("Watershed Segmentation")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    props = regionprops(labels, intensity_image=image)
    regions_info = []
    for region in props:
        perimeter = region.perimeter if region.perimeter != 0 else 1
        circularity = 4 * np.pi * region.area / (perimeter ** 2)
        info = {
            "label": region.label,
            "area": region.area,
            "centroid": region.centroid,
            "bbox": region.bbox,
            "mean_intensity": region.mean_intensity,
            "circularity": circularity,
            "major_axis_length": region.major_axis_length,
            "minor_axis_length": region.minor_axis_length
        }
        regions_info.append(info)
    return labels, regions_info

def print_region_metrics(region_data, labels):
    print(f"{'Label':<8} {'Area':<8} {'Centrality':<12} {'Solidity':<10}")
    print("-" * 42)

    for region in region_data:
        label = region["label"]
        coords = np.column_stack(np.where(labels == label))
        if coords.shape[0] < 3:
            continue

        centroid = np.array(region["centroid"])
        dists = np.linalg.norm(coords - centroid, axis=1)
        centrality = np.std(dists)

        try:
            hull = ConvexHull(coords)
            convex_area = hull.volume
            solidity = coords.shape[0] / convex_area if convex_area > 0 else 0
        except Exception:
            solidity = 0

        print(f"{label:<8} {region['area']:<8} {centrality:<12.2f} {solidity:<10.2f}")
    

def auto_select_uterus(labels, region_data, image_shape, centrality_thresh=20,
                      solidity_thresh=0.7, area_thresh=2000, image=None, visualize=True, debug=False):
    candidates = []
    
    if debug:
        print(f"Analyzing {len(region_data)} regions with thresholds: centrality < {centrality_thresh}, solidity > {solidity_thresh}, area > {area_thresh}")
    
    for region in region_data:
        
        region_mask = (labels == region["label"])
        
        # ===  centrality ===
        coords = np.column_stack(np.where(region_mask))
        centroid = np.array(region["centroid"])
        dists = np.linalg.norm(coords - centroid, axis=1)
        centrality = np.std(dists)
        
        # ===  solidity ===
        try:
            if coords.shape[0] < 3:
                solidity = 0
            else:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(coords)
                convex_area = hull.volume
                solidity = coords.shape[0] / convex_area if convex_area > 0 else 0
        except Exception as e:
            if debug:
                print(f"Error calculating solidity for region {region['label']}: {e}")
            solidity = 0  
        
        # ===  area ===
        area = region.get("area", coords.shape[0])
        if debug:
            print(f"Region {region['label']}: centrality={centrality:.2f}, solidity={solidity:.2f}, area={area}")
        
        if centrality > centrality_thresh or solidity < solidity_thresh or area < area_thresh:
            if debug:
                print(f"  Rejected: centrality={centrality:.2f}>{centrality_thresh} or solidity={solidity:.2f}<{solidity_thresh} or area={area}<{area_thresh}")
            continue
        
        if debug:
            print(f"  Accepted as candidate")
        region["centrality"] = centrality
        region["solidity"] = solidity
        candidates.append(region)
    
    if not candidates:
        if debug or len(region_data) > 0:  
            print("[Uterus] No suitable uterus region found.")
        return np.zeros(image_shape, dtype=bool)
    
    # Sort by centrality in ascending order (select the smallest)
    candidates_sorted = sorted(candidates, key=lambda r: r["centrality"])
    selected = candidates_sorted[0]
    area = selected.get("area", 0)
    if debug:
        print(f"Selected region {selected['label']} with centrality={selected['centrality']:.2f}, solidity={selected['solidity']:.2f}, area={area}")
    
    uterus_mask = (labels == selected["label"])
    
    if visualize:
        plt.figure(figsize=(6, 6))
        if image is not None:
            plt.imshow(image, cmap='gray')
            plt.imshow(uterus_mask, cmap='Reds', alpha=0.5)
        else:
            plt.imshow(uterus_mask, cmap='gray')
        area = selected.get("area", 0)
        plt.title(f"Selected Uterus (Label {selected['label']}, Centrality={selected['centrality']:.2f}, Solidity={selected['solidity']:.2f}, Area={area})")
        plt.axis('off')
        plt.show()
    
    return uterus_mask

import os
from glob import glob

def main_batch_images(folder_path,
                      bladder_intensity_thresh=0.7,
                      bladder_area_thresh=1000,
                      seed_area_thresh=500,
                      seed_aspect_ratio=(0.3, 3.0),
                      uterus_centrality_thresh=20,
                      uterus_solidity_thresh=0.9,
                      visualize=False):
    valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.bmp'}
    image_files = sorted([
        f for f in glob(os.path.join(folder_path, "*"))
        if os.path.splitext(f)[1].lower() in valid_exts
    ])
    if not image_files:
        print("[Error] No valid image files found.")
        return

    result_folder = folder_path.rstrip("/\\") + "_result"
    os.makedirs(result_folder, exist_ok=True)
    
    for image_path in image_files:
        # print(f"\n[Processing] {os.path.basename(image_path)}")
        try:
            image = load_mri_image(image_path)
            bladder_mask = find_bladder_region(
                image,
                intensity_thresh=bladder_intensity_thresh,
                area_thresh=bladder_area_thresh,
                visualize=visualize,
            )
            seeds = generate_seeds(
                image,
                area_thresh=seed_area_thresh,
                aspect_ratio_range=seed_aspect_ratio,
                visualize=visualize,
            )
            labels, region_data = segment_uterus(image, seeds, visualize=visualize)
            uterus_mask = auto_select_uterus(
                labels,
                region_data,
                image_shape=image.shape,
                centrality_thresh=uterus_centrality_thresh,
                solidity_thresh=uterus_solidity_thresh,
                image=image,
                visualize=visualize,
                debug=False
            )
            final_mask = uterus_mask.copy()
            overlap = np.logical_and(uterus_mask, bladder_mask)
            if np.any(overlap):
                final_mask[bladder_mask] = 0
                print("[Final] Overlap detected. Bladder region subtracted.")
            else:
                print("[Final] No overlap with bladder mask.")

            # save & visualize
            save_name = os.path.splitext(os.path.basename(image_path))[0] + "_overlay.png"
            save_path = os.path.join(result_folder, save_name)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image, cmap='gray')
            ax.imshow(final_mask, cmap='Blues', alpha=0.5)
            ax.set_title(f"Final Result: {os.path.basename(image_path)}")
            ax.axis('off')
            plt.tight_layout()
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
            plt.figure(figsize=(6, 6))
            plt.imshow(image, cmap='gray')
            plt.imshow(final_mask, cmap='Blues', alpha=0.5)
            plt.title(f"Final Result: {os.path.basename(image_path)}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[Error] Failed to process {image_path}: {e}")