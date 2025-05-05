# Uterus Seed Generation via Morphological Filtering

This script performs seed point generation for uterus segmentation based on morphological processing. It includes automatic batch bladder removal and is designed for MRI image slices.

### How to use

You can run the function with:

```python
main_batch_images(
    folder_path="youfolderpath/",
    bladder_intensity_thresh=0.7,
    bladder_area_thresh=1000,
    seed_area_thresh=500,
    seed_aspect_ratio=(0.3, 3.0),
    uterus_centrality_thresh=20,
    uterus_solidity_thresh=0.5,
    visualize=False  
)
