import os 
import numpy as np 
import nibabel as nib
from scipy.ndimage import zoom

folder = r"C:\Users\srbel\BIOE484_Sp25_Final_Project\NIFTIES" #Put in your folder 
nifti_files = [file for file in os.listdir(folder)]
full_paths = [os.path.join(folder,f) for f in nifti_files]

target_shape = (335, 335, 40) #Most common dimension among NIFTI files

print("Processing images...")
resampled_arrays = []
for path in full_paths:
    img = nib.load(path)
    data = img.get_fdata()
    
    zoom_factors = [target_shape[i] / data.shape[i] for i in range(3)] #Finds zoom factors for each dimension 

    resampled_data = zoom(data, zoom_factors, order=1)  # resamples the image and uses order=1 for linear interpolation
    resampled_arrays.append(resampled_data)
    print(f"Resampled {os.path.basename(path)} from {data.shape} to {resampled_data.shape}")

# stacks and averages 
stacked = np.stack(resampled_arrays)
mean_img = np.mean(stacked, axis=0)

# saves result
affine = nib.load(full_paths[0]).affine
avg_nifti = nib.Nifti1Image(mean_img, affine)
nib.save(avg_nifti, "average_mouse_atlas.nii.gz")
print("\nAverage atlas saved as 'average_mouse_atlas.nii.gz'")

