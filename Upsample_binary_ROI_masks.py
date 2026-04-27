import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import os

'''
This file is used to upsample the binary ROI masks, in this case CA1 (head and tail excluded) obtained by ASHS, to 0.2mm isotropic resolution. 
'''

def load_nifti(filepath):
    img = nib.load(filepath)
    return img

def crop_to_roi(ashs_img):
    data = ashs_img.get_fdata().astype(np.uint8)
    affine = ashs_img.affine.copy()

    # Bounding box of nonzero region
    nonzero_coords = np.array(np.nonzero(data))
    min_coords = nonzero_coords.min(axis=1)
    max_coords = nonzero_coords.max(axis=1) + 1

    cropped_data = data[
        min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1],
        min_coords[2]:max_coords[2]
    ]

    # Update affine origin to new bounding box start
    new_affine = affine.copy()
    new_affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ min_coords

    return cropped_data, new_affine

def upsample_ASHS(cropped_data, affine):
    # Derive original voxel sizes from affine (more robust than header after cropping)
    original_voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    zoom_factors = [original_voxel_sizes[i] / 0.2 for i in range(3)]

    print(f"Original voxel sizes: {original_voxel_sizes}")
    print(f"Zoom factors: {zoom_factors}")

    upsampled_data = zoom(cropped_data, zoom_factors, order=0)  # nearest neighbor for masks

    # Scale affine columns to new voxel size
    new_affine = affine.copy()
    scaling_factors = [0.2 / original_voxel_sizes[i] for i in range(3)]
    for i in range(3):
        new_affine[:3, i] *= scaling_factors[i]

    print(f"New data shape: {upsampled_data.shape}")

    new_img = nib.Nifti1Image(upsampled_data, new_affine)
    print(f"New voxel sizes: {new_img.header.get_zooms()}")

    return new_img

# --- Paths ---
input_dir = "Path/to/subject_folders"
sides = ["LH", "RH"] 

subjects = os.listdir(input_dir_1)
subjects.sort()

for sub in subjects:
    for side in sides:
        ashs_path = os.path.join(input_dir_1, sub, f"r{sub}_mask_CA1_{side}.nii")
        try:
            ashs_img = load_nifti(ashs_path)
            print(f"Now cropping and upsampling {sub} {side}...")

            # Step 1: Crop to ROI bounding box
            cropped_data, cropped_affine = crop_to_roi(ashs_img)
            print(f"Cropped shape: {cropped_data.shape}")

            # Step 2: Upsample the cropped ROI
            ashs_img_ups = upsample_ASHS(cropped_data, cropped_affine)

            # Save
            output_dir = os.path.join(input_dir, sub, "upsampled_data")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{sub}_CA1_{side}_HB_upsampled_mask.nii.gz")

            if os.path.exists(output_path):
                print(f"Replacing existing file: {output_path}")
                os.remove(output_path)

            nib.save(ashs_img_ups, output_path)
            print(f"{sub} {side} complete.")

        except FileNotFoundError:
            print(f"File not found: {ashs_path}")
        except Exception as e:
            print(f"Error processing {sub} {side}: {str(e)}")

print("All subjects processed!")
