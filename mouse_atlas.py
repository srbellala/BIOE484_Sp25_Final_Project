import os 
import numpy as np 
import nibabel as nib


folder = " " #Put in your folder 
nifti_files = [file for file in os.listdir(folder)]
full_paths = [os.path.join(folder,f) for f in nifti_files]

arrays = [nib.load(file).get_fdata() for file in full_paths]

stacked = np.stack(arrays)
mean_img = np.mean(stacked, axis = 0)

affine = nib.load(full_paths[0]).affine
avg_nifti = nib.Nifti1Image(mean_img,affine)
nib.save(avg_nifti, "average_mouse_atlas.nii.gz")

