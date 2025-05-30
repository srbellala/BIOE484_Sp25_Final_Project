import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pydicom
from skimage.filters import threshold_otsu, sobel
from skimage.morphology import binary_opening, binary_closing, disk, remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import imageio
from skimage import exposure, measure
import matplotlib.patches as patches
from skimage.morphology import binary_dilation, skeletonize
from scipy.ndimage import convolve, distance_transform_edt, binary_fill_holes, gaussian_filter
from skimage.segmentation import watershed
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import nibabel as nib

@dataclass
class SegmentationConfig:
    """Configuration parameters for the segmentation pipeline."""
    # Output directories
    output_dir: str = "output_plots"
    final_output_dir: str = "output_final"
    
    # Segmentation parameters
    min_area: int = 300
    max_area: int = 80000
    y_branch_threshold: int = 3
    ellipse_ecc_threshold: float = 0.95
    centrality_dilation_px: int = 10
    centrality_min_obj_area: int = 50
    
    # Preprocessing parameters
    gaussian_sigma: float = 0.8
    clahe_clip_limit: float = 0.03
    clahe_kernel_size: Optional[int] = None
    
    # Thresholding parameters
    percentile_thresholds: List[int] = (40, 45, 50, 55, 60, 65, 70)
    edge_percentiles: List[int] = (40, 45, 50, 55, 60)
    
    def __post_init__(self):
        """Initialize derived parameters."""
        if self.clahe_kernel_size is None:
            self.clahe_kernel_size = 8  # Will be adjusted based on image size

class UterusSegmenter:
    """Optimized Mouse Uterus Segmenter with Method Tracking"""
    
    def __init__(self, config: Optional[SegmentationConfig] = None):
        """Initialize the segmenter with configuration."""
        self.config = config or SegmentationConfig()
        self.logger = self._setup_logging()
        self._create_output_dirs()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('segmentation.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _create_output_dirs(self):
        """Create output directories if they don't exist."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.final_output_dir).mkdir(parents=True, exist_ok=True)

    def load_frames(self, filepath):
        """Load and normalize frames from DICOM file"""
        ds = pydicom.dcmread(filepath)
        arr = ds.pixel_array.astype(np.float32)
        
        # Apply VOI LUT if available
        try:
            from pydicom.pixel_data_handlers.util import apply_voi_lut
            if arr.ndim == 3:
                arr = np.stack([apply_voi_lut(arr[i], ds) for i in range(arr.shape[0])])
            else:
                arr = apply_voi_lut(arr, ds)
        except:
            pass  # Use raw pixel array
        
        # Normalize to [0,1]
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        
        for i in range(arr.shape[0]):
            frame = arr[i]
            p0_5, p99_5 = np.percentile(frame, [0.5, 99.5])
            frame = np.clip(frame, p0_5, p99_5)
            arr[i] = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        
        return arr

    def preprocess_image(self, img):
        """Enhanced preprocessing with validation and adaptive parameters."""
        try:
            if img.size == 0:
                raise ValueError("Empty image provided for preprocessing")
            
            # Ensure image is in [0,1] range
            if img.max() > 1.0 or img.min() < 0.0:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            # Adaptive CLAHE kernel size based on image dimensions
            kernel_size = min(img.shape[0], img.shape[1]) // self.config.clahe_kernel_size
            
            # CLAHE enhancement
            enhanced = exposure.equalize_adapthist(
                img, 
                clip_limit=self.config.clahe_clip_limit,
                kernel_size=kernel_size
            )
            
            # Denoising with adaptive parameters
            denoised = gaussian_filter(enhanced, sigma=self.config.gaussian_sigma)
            
            # Edge detection
            edges = sobel(enhanced)
            
            self.logger.info("Preprocessing completed successfully")
            return denoised, enhanced, edges
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise

    def generate_candidate_masks(self, img):
        """Generate multiple candidate masks using different thresholding methods with improved error handling."""
        try:
            denoised, enhanced, edges = self.preprocess_image(img)
            masks = []
            methods_used = []
            method_details = {}
            
            # Otsu thresholding with adaptation
            try:
                thresh_otsu = threshold_otsu(denoised)
                img_mean = np.mean(denoised)
                adapt_factor = 0.7 if img_mean < 0.3 else (1.2 if img_mean > 0.7 else 0.9)
                final_thresh = thresh_otsu * adapt_factor
                mask_otsu = denoised > final_thresh
                masks.append(mask_otsu)
                methods_used.append("adaptive_otsu")
                method_details["adaptive_otsu"] = {
                    "base_threshold": float(thresh_otsu),
                    "adaptation_factor": float(adapt_factor),
                    "final_threshold": float(final_thresh),
                    "image_mean": float(img_mean)
                }
            except Exception as e:
                self.logger.warning(f"Adaptive Otsu thresholding failed: {str(e)}")
                method_details["adaptive_otsu"] = {"error": str(e)}
            
            # Percentile thresholding with configurable thresholds
            percentile_methods = []
            for p in self.config.percentile_thresholds:
                try:
                    thresh_p = np.percentile(denoised, p)
                    mask_p = denoised > thresh_p
                    mask_coverage = np.mean(mask_p)
                    if 0.05 <= mask_coverage <= 0.85:
                        masks.append(mask_p)
                        method_name = f"percentile_{p}"
                        methods_used.append(method_name)
                        percentile_methods.append({
                            "percentile": p,
                            "threshold": float(thresh_p),
                            "coverage": float(mask_coverage),
                            "used": True
                        })
                    else:
                        percentile_methods.append({
                            "percentile": p,
                            "threshold": float(thresh_p),
                            "coverage": float(mask_coverage),
                            "used": False,
                            "reason": "coverage outside valid range"
                        })
                except Exception as e:
                    self.logger.warning(f"Percentile thresholding failed for {p}: {str(e)}")
                    percentile_methods.append({
                        "percentile": p,
                        "error": str(e),
                        "used": False
                    })
            method_details["percentile_thresholding"] = percentile_methods
            
            # Edge-based masks with configurable parameters
            edge_methods = []
            for ep in self.config.edge_percentiles:
                try:
                    if not np.isclose(edges.max(), edges.min()):
                        edge_thresh = np.percentile(edges, ep)
                        mask_edges = binary_fill_holes(binary_dilation(edges > edge_thresh, disk(1)))
                        edge_coverage = np.mean(mask_edges)
                        if 0.005 <= edge_coverage <= 0.75:
                            masks.append(mask_edges)
                            method_name = f"edge_based_{ep}"
                            methods_used.append(method_name)
                            edge_methods.append({
                                "percentile": ep,
                                "threshold": float(edge_thresh),
                                "coverage": float(edge_coverage),
                                "used": True
                            })
                        else:
                            edge_methods.append({
                                "percentile": ep,
                                "threshold": float(edge_thresh),
                                "coverage": float(edge_coverage),
                                "used": False,
                                "reason": "coverage outside valid range"
                            })
                    else:
                        edge_methods.append({
                            "percentile": ep,
                            "used": False,
                            "reason": "edges have no variation"
                        })
                except Exception as e:
                    self.logger.warning(f"Edge-based mask generation failed for {ep}: {str(e)}")
                    edge_methods.append({
                        "percentile": ep,
                        "error": str(e),
                        "used": False
                    })
            method_details["edge_based"] = edge_methods
            
            # Combine masks using majority voting with improved logic
            if not masks:
                self.logger.warning("No valid masks generated, using fallback threshold")
                fallback_thresh = np.percentile(denoised, 35)
                combined = denoised > fallback_thresh
                methods_used = ["fallback_percentile_35"]
                method_details["final_combination"] = {
                    "method": "fallback",
                    "threshold": float(fallback_thresh),
                    "reason": "no valid masks generated"
                }
            else:
                if len(masks) >= 3:
                    required_votes = len(masks) // 2 + 1
                    combined = np.sum(masks, axis=0) >= required_votes
                    combination_method = f"majority_voting_{required_votes}_of_{len(masks)}"
                elif len(masks) == 2:
                    combined = np.sum(masks, axis=0) >= 2
                    combination_method = "unanimous_2_of_2"
                else:
                    combined = masks[0]
                    combination_method = "single_method"
                
                method_details["final_combination"] = {
                    "method": combination_method,
                    "total_masks": len(masks),
                    "methods_combined": methods_used.copy()
                }
            
            self.logger.info(f"Generated {len(masks)} candidate masks using {len(methods_used)} methods")
            return combined, method_details
            
        except Exception as e:
            self.logger.error(f"Mask generation failed: {str(e)}")
            raise

    def watershed_refinement(self, mask, img):
        """Apply watershed segmentation for better separation"""
        if not np.any(mask):
            return mask
        
        distance = distance_transform_edt(mask)
        
        # Find local maxima for markers
        markers = label(distance > 0.5 * distance.max())
        
        if np.max(markers) == 0:
            return mask
        
        # Apply watershed
        gradient = sobel(gaussian_filter(img, sigma=0.5))
        labels_ws = watershed(gradient, markers, mask=mask, compactness=0.001)
        return labels_ws > 0

    def morphological_cleanup(self, mask):
        """Apply morphological operations to clean up the mask"""
        if not np.any(mask):
            return mask
        
        clean_mask = remove_small_objects(mask, min_size=max(10, self.config.centrality_min_obj_area // 2))
        clean_mask = remove_small_holes(clean_mask, area_threshold=200)
        clean_mask = binary_opening(clean_mask, disk(1))
        clean_mask = binary_closing(clean_mask, disk(2))
        
        return clean_mask

    def filter_by_shape(self, mask):
        """Filter regions based on shape characteristics"""
        filtered = np.zeros_like(mask, dtype=bool)
        labeled = label(mask)
        
        for prop in regionprops(labeled):
            if not (self.config.min_area <= prop.area <= self.config.max_area):
                continue
            
            region_mask = (labeled == prop.label)
            
            # Check eccentricity
            ellipse_cond = prop.eccentricity < self.config.ellipse_ecc_threshold
            
            # Check for Y-shaped structure (branching)
            skeleton = skeletonize(region_mask)
            kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
            neigh_count = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
            branch_points = skeleton & (neigh_count >= self.config.y_branch_threshold)
            y_cond = np.any(branch_points)
            
            if ellipse_cond or y_cond:
                filtered |= region_mask
        
        return filtered

    def filter_by_centrality(self, mask, roi_coords_xyxy, image_shape):
        """Keep only regions within ROI and near the largest component"""
        if not np.any(mask) or not roi_coords_xyxy:
            return mask
        
        # Create ROI mask
        roi_mask = np.zeros(image_shape, dtype=bool)
        x0, y0, x1, y1 = roi_coords_xyxy
        r0, c0 = max(0, int(y0)), max(0, int(x0))
        r1, c1 = min(image_shape[0], int(y1)), min(image_shape[1], int(x1))
        
        if r1 <= r0 or c1 <= c0:
            return np.zeros(image_shape, dtype=bool)
        
        roi_mask[r0:r1, c0:c1] = True
        
        # Keep only mask parts within ROI
        mask_in_roi = mask & roi_mask
        if not np.any(mask_in_roi):
            return mask_in_roi
        
        # Find largest component in ROI
        labeled = label(mask_in_roi)
        props = regionprops(labeled)
        if not props:
            return np.zeros(image_shape, dtype=bool)
        
        largest_prop = max(props, key=lambda p: p.area)
        main_mask = (labeled == largest_prop.label)
        
        # Include nearby components
        if self.config.centrality_dilation_px > 0:
            dilated_main = binary_dilation(main_mask, disk(self.config.centrality_dilation_px))
            final_mask = mask_in_roi & dilated_main
        else:
            final_mask = main_mask
        
        # Remove small fragments
        if np.any(final_mask):
            final_mask = remove_small_objects(final_mask, min_size=self.config.centrality_min_obj_area)
        
        return final_mask

    def auto_detect_roi(self, img):
        """Automatically detect ROI based on image content"""
        denoised, _, edges = self.preprocess_image(img)
        roi_masks = []
        
        # Edge-based ROI
        if not np.isclose(edges.max(), edges.min()):
            edge_thresh = np.percentile(edges, 65)
            edge_mask = binary_closing(binary_dilation(edges > edge_thresh, disk(7)), disk(15))
            roi_masks.append(edge_mask)
        
        # Otsu-based ROI
        try:
            otsu_thresh = threshold_otsu(denoised)
            roi_masks.append(denoised > (otsu_thresh * 0.55))
        except:
            pass
        
        # Percentile-based ROI
        roi_masks.append(denoised > np.percentile(denoised, 50))
        
        # Combine valid masks
        valid_masks = [m for m in roi_masks if m is not None and np.any(m)]
        if valid_masks:
            rough_roi = np.logical_or.reduce(valid_masks)
        else:
            rough_roi = denoised > np.percentile(denoised, 40)
        
        # Clean up ROI
        rough_roi = binary_closing(rough_roi, disk(10))
        rough_roi = remove_small_objects(rough_roi, min_size=img.size * 0.005)
        
        h, w = img.shape
        if not np.any(rough_roi):
            return h//4, w//4, 3*h//4, 3*w//4  # Fallback center ROI
        
        # Get bounding box of largest component
        labeled_roi = label(rough_roi)
        props = regionprops(labeled_roi)
        if not props:
            return h//4, w//4, 3*h//4, 3*w//4
        
        largest_prop = max(props, key=lambda p: p.area)
        r0, c0, r1, c1 = largest_prop.bbox
        
        # Add padding
        padding = int(min(h, w) * 0.05)
        return (max(0, r0 - padding), max(0, c0 - padding), 
                min(h, r1 + padding), min(w, c1 + padding))

    def segment_frame(self, img, manual_roi_xyxy=None):
        """Frame segmentation without active contour refinement."""
        try:
            # Generate initial mask
            initial_mask, method_details = self.generate_candidate_masks(img)
            
            # Apply watershed refinement
            watershed_mask = self.watershed_refinement(initial_mask, img)
            
            # Apply morphological cleanup
            cleaned_mask = self.morphological_cleanup(watershed_mask)
            
            # Filter by shape
            shape_filtered = self.filter_by_shape(cleaned_mask)
            
            # Filter by centrality if ROI is provided
            if manual_roi_xyxy is not None:
                final_mask = self.filter_by_centrality(shape_filtered, manual_roi_xyxy, img.shape)
            else:
                # Auto-detect ROI if not provided
                roi = self.auto_detect_roi(img)
                final_mask = self.filter_by_centrality(shape_filtered, roi, img.shape)
            
            self.logger.info("Frame segmentation completed successfully (no active contour)")
            return final_mask, method_details
        except Exception as e:
            self.logger.error(f"Frame segmentation failed: {str(e)}")
            raise

    def visualize(self, img, mask, frame_idx=None, roi_xyxy=None, show_analysis=False, method_report=None):
        """Visualize segmentation results"""
        if show_analysis:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()
            
            # Original image
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Preprocessed images
            denoised, enhanced, edges = self.preprocess_image(img)
            axes[1].imshow(enhanced, cmap='gray')
            axes[1].set_title('CLAHE Enhanced')
            axes[1].axis('off')
            
            axes[2].imshow(edges, cmap='hot')
            axes[2].set_title('Sobel Edges')
            axes[2].axis('off')
            
            # ROI visualization
            axes[3].imshow(img, cmap='gray')
            axes[3].set_title('ROI Detection')
            if roi_xyxy:
                x0, y0, x1, y1 = roi_xyxy
                rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, 
                                       linewidth=1, edgecolor='yellow', 
                                       facecolor='none', linestyle=':')
                axes[3].add_patch(rect)
            axes[3].axis('off')
            
            # Segmentation overlay
            axes[4].imshow(img, cmap='gray')
            if np.any(mask):
                axes[4].imshow(mask, alpha=0.4, cmap='Reds')
            title = 'Segmentation Overlay'
            if method_report and 'final_combination' in method_report:
                methods_used = method_report['final_combination'].get('methods_combined', [])
                title += f'\nMethods: {len(methods_used)}'
            axes[4].set_title(title)
            axes[4].axis('off')
            
            # Final result with contours
            axes[5].imshow(img, cmap='gray')
            if np.any(mask):
                contours = measure.find_contours(mask, 0.5)
                for contour in contours:
                    axes[5].plot(contour[:, 1], contour[:, 0], 'lime', linewidth=1.5)
            if roi_xyxy:
                x0, y0, x1, y1 = roi_xyxy
                rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, 
                                       linewidth=1.5, edgecolor='cyan', 
                                       facecolor='none', linestyle='--')
                axes[5].add_patch(rect)
            title = f'Final Result (Frame {frame_idx})' if frame_idx is not None else 'Final Result'
            axes[5].set_title(title)
            axes[5].axis('off')
            
            plt.tight_layout()
            
            # Print method report if available
            if method_report:
                print(f"\n=== Method Report for Frame {frame_idx} ===")
                if 'final_combination' in method_report:
                    methods_used = method_report['final_combination'].get('methods_combined', [])
                    print(f"Methods used: {', '.join(methods_used)}")
                    print(f"Final combination: {method_report['final_combination']['method']}")
        else:
            # Simple visualization
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img, cmap='gray')
            
            if np.any(mask):
                contours = measure.find_contours(mask, 0.5)
                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], 'red', linewidth=1.5)
                ax.imshow(mask, alpha=0.3, cmap='Reds')
            else:
                ax.text(img.shape[1]*0.5, img.shape[0]*0.5, 'No Structure Detected', 
                        ha='center', va='center', color='white', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.5", fc="red", alpha=0.7))
            
            if roi_xyxy:
                x0, y0, x1, y1 = roi_xyxy
                rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, 
                                       linewidth=1.5, edgecolor='blue', 
                                       facecolor='none', linestyle='--')
                ax.add_patch(rect)
            
            title = f"Segmentation: Frame {frame_idx}" if frame_idx is not None else "Segmentation"
            if method_report and 'final_combination' in method_report:
                methods_used = method_report['final_combination'].get('methods_combined', [])
                title += f" ({len(methods_used)} methods)"
            ax.set_title(title, fontsize=12)
            ax.axis('off')
        
        plt.show()

    def process_file(self, input_file, output_dir, manual_rois_xyxy=None, 
                     display=False, slice_idx=None, show_analysis=False,
                     volume_start=None, volume_end=None):
        """Process entire DICOM file"""
        os.makedirs(output_dir, exist_ok=True)
        frames = self.load_frames(input_file)
        
        # Always collect all masks for volume rendering
        all_masks = []
        for i in range(frames.shape[0]):
            img = frames[i]
            manual_roi = manual_rois_xyxy.get(i) if manual_rois_xyxy else None
            mask, _ = self.segment_frame(img, manual_roi)
            all_masks.append(mask)
        
        # For output/metrics, only process the requested slice(s)
        indices = [slice_idx] if slice_idx is not None else range(frames.shape[0])
        results = []
        method_reports = {}
        
        for i in indices:
            if not (0 <= i < frames.shape[0]):
                print(f"Slice index {i} out of bounds. Skipping.")
                continue
            
            img = frames[i]
            manual_roi = manual_rois_xyxy.get(i) if manual_rois_xyxy else None
            
            # Determine ROI for visualization
            if manual_roi:
                roi_for_viz = manual_roi
            else:
                r0, c0, r1, c1 = self.auto_detect_roi(img)
                roi_for_viz = (c0, r0, c1, r1)
            
            # Segment frame (already done above, but for metrics/output, re-use)
            mask = all_masks[i]
            method_report = None  # Optionally, you could re-run or store method_report if needed
            method_reports[i] = method_report
            
            # Save results
            mask_path = os.path.join(output_dir, f"frame_{i:03d}_mask.png")
            imageio.imsave(mask_path, (mask.astype(np.uint8) * 255))
            
            overlay_path = os.path.join(output_dir, f"frame_{i:03d}_overlay.png")
            fig, ax = plt.subplots(figsize=(img.shape[1]/100, img.shape[0]/100), dpi=100)
            ax.imshow(img, cmap='gray')
            if np.any(mask):
                contours = measure.find_contours(mask, 0.5)
                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], 'red', linewidth=0.8)
                ax.imshow(mask, alpha=0.25, cmap='Reds')
            if roi_for_viz:
                x0, y0, x1, y1 = roi_for_viz
                rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, 
                                       linewidth=0.8, edgecolor='blue', 
                                       facecolor='none', linestyle='--')
                ax.add_patch(rect)
            ax.axis('off')
            title = f'Frame {i}'
            # method_report is None here, but you can add logic to store it if needed
            ax.set_title(title, fontsize=8)
            plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close()
            
            # Save method report for this frame (if available)
            if method_report:
                method_path = os.path.join(output_dir, f"frame_{i:03d}_methods.json")
                with open(method_path, 'w') as f:
                    json.dump(method_report, f, indent=2)
            
            if display:
                self.visualize(img, mask, frame_idx=i, roi_xyxy=roi_for_viz, 
                              show_analysis=show_analysis, method_report=method_report)
            
            # Calculate metrics
            area = np.sum(mask)
            detected = np.any(mask)
            num_contours = 0
            contour_length = 0
            perimeter = 0
            
            if detected:
                contours = measure.find_contours(mask, 0.5)
                num_contours = len(contours)
                contour_length = sum(len(c) for c in contours)
                try:
                    perimeter = measure.perimeter(mask)
                except:
                    perimeter = 0
            
            result = {
                'frame': i, 'detected': detected, 'area': area,
                'num_contours': num_contours, 'contour_length': contour_length,
                'perimeter': perimeter
            }
            results.append(result)
        
        # Save comprehensive method summary
        method_summary_path = os.path.join(output_dir, "method_summary.json")
        with open(method_summary_path, 'w') as f:
            json.dump(method_reports, f, indent=2)
        
        # Print summary with method information
        print(f"\n=== Segmentation Summary ===")
        detected_frames = sum(1 for r in results if r['detected'])
        print(f"Processed frames: {len(results)}")
        print(f"Detected structures: {detected_frames}")
        print(f"Detection rate: {detected_frames/len(results)*100:.1f}%")
        
        if detected_frames > 0:
            areas = [r['area'] for r in results if r['detected']]
            print(f"Mean area: {np.mean(areas):.1f} ± {np.std(areas):.1f}")
        
        # --- Volume Rendering with matplotlib ---
        if len(all_masks) > 1:
            try:
                # Determine the range for volume rendering
                if volume_start is None:
                    volume_start = 0
                if volume_end is None or volume_end > len(all_masks):
                    volume_end = len(all_masks)
                selected_masks = all_masks[volume_start:volume_end]
                if len(selected_masks) > 1:
                    mask_volume = np.stack(selected_masks, axis=0)
                    verts, faces, normals, values = measure.marching_cubes(mask_volume, level=0.5)
                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(111, projection='3d')
                    mesh = Poly3DCollection(verts[faces], alpha=0.7)
                    mesh.set_facecolor('red')
                    ax.add_collection3d(mesh)
                    ax.set_xlim(0, mask_volume.shape[0])
                    ax.set_ylim(0, mask_volume.shape[1])
                    ax.set_zlim(0, mask_volume.shape[2])
                    ax.set_xlabel('Z (slice)')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('X')
                    plt.tight_layout()
                    plt.show()
                    # --- Export mask as NIfTI ---
                    nifti_path = os.path.join(output_dir, 'segmentation_mask.nii.gz')
                    nii_img = nib.Nifti1Image(mask_volume.astype(np.uint8), affine=np.eye(4))
                    nib.save(nii_img, nifti_path)
                    print(f'NIfTI mask saved to: {nifti_path}')
            except Exception as e:
                print(f"Volume rendering or NIfTI export failed: {e}")
        # --- End Volume Rendering ---
        
        return results


def parse_roi_xyxy(roi_str):
    """Parse ROI string in format 'x0,y0,x1,y1'"""
    try:
        coords = [int(x) for x in roi_str.split(',')]
        if len(coords) != 4:
            raise ValueError("ROI must be x0,y0,x1,y1")
        return tuple(coords)
    except Exception as e:
        raise ValueError(f"Invalid ROI string '{roi_str}': {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Uterus Segmentation with Method Tracking")
    parser.add_argument("--input_file", required=True, help="DICOM file path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--roi", help="Manual ROI for all frames: 'x0,y0,x1,y1'")
    parser.add_argument("--roi_file", help="CSV file with ROIs per frame: frame_idx,x0,y0,x1,y1")
    parser.add_argument("--display", action="store_true", help="Display results")
    parser.add_argument("--show_analysis", action="store_true", help="Show analysis steps")
    parser.add_argument("--slice_idx", type=int, help="Process specific frame index")
    parser.add_argument("--min_area", type=int, default=300, help="Minimum area")
    parser.add_argument("--max_area", type=int, default=80000, help="Maximum area")
    parser.add_argument("--centrality_dilation", type=int, default=10, help="Centrality dilation")
    parser.add_argument("--centrality_min_obj", type=int, default=50, help="Min object area")
    parser.add_argument("--volume_start", type=int, help="First frame to include in volume render (default: 0)")
    parser.add_argument("--volume_end", type=int, help="Last frame (exclusive) to include in volume render (default: all)")
    
    args = parser.parse_args()
    
    # Parse ROI arguments
    manual_rois = {}
    if args.roi_file:
        try:
            with open(args.roi_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 5:
                        try:
                            idx = int(parts[0])
                            coords = tuple(map(int, parts[1:5]))
                            manual_rois[idx] = coords
                        except ValueError:
                            print(f"Skipping invalid ROI line: {line.strip()}")
        except FileNotFoundError:
            print(f"ROI file not found: {args.roi_file}")
    elif args.roi:
        try:
            parsed_roi = parse_roi_xyxy(args.roi)
            if args.slice_idx is not None:
                manual_rois = {args.slice_idx: parsed_roi}
            else:
                manual_rois = {i: parsed_roi for i in range(1000)}
        except ValueError as e:
            print(f"Invalid --roi: {e}")
    
    # Create configuration
    config = SegmentationConfig(
        min_area=args.min_area,
        max_area=args.max_area,
        centrality_dilation_px=args.centrality_dilation,
        centrality_min_obj_area=args.centrality_min_obj
    )
    
    # Create segmenter with configuration
    segmenter = UterusSegmenter(config=config)
    
    try:
        results = segmenter.process_file(
            args.input_file, args.output_dir,
            manual_rois_xyxy=manual_rois if manual_rois else None,
            display=args.display, slice_idx=args.slice_idx,
            show_analysis=args.show_analysis,
            volume_start=args.volume_start, volume_end=args.volume_end
        )
        print("\nProcessing Complete.")
        print(f"\nOutput files saved to: {args.output_dir}")
        print("- Individual method reports: frame_XXX_methods.json")
        print("- Comprehensive method summary: method_summary.json")
        print("- Mask files: frame_XXX_mask.png")
        print("- Overlay images: frame_XXX_overlay.png")
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()