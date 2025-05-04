"""
Main script for uterus seed generation using Frangi filter and symmetric region analysis.
Suitable for MRI slices where the abdominal cavity contains signal.
"""

from frangi_pipeline.enhancement import frangi_vessel_enhancement
from frangi_pipeline.seed_extraction import extract_symmetric_seeds_from_region
from utils.image_loader import load_mri_image


def main():
    # Load grayscale float image from file
    img_path = "data/2.png"  # Update path as needed
    image = load_mri_image(img_path)

    # Step 1: Enhance vessels using Frangi filter
    vessel_mask = frangi_vessel_enhancement(
        image,
        visualize=True,
        sigmas=(1, 2, 3),
        threshold=0.2,
        closing_radius=5
    )

    # Step 2: Extract symmetric seed points
    seed1, seed2 = extract_symmetric_seeds_from_region(
        vessel_mask,
        visualize=True
    )

    print("Seed 1:", seed1)
    print("Seed 2:", seed2)


if __name__ == "__main__":
    main()
