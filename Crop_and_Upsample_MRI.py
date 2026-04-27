import os
import nibabel as nib
from nibabel import Nifti1Image
from nibabel.processing import resample_from_to
import numpy as np

"""
Since we want to use our layered masks to extract depth-dependent data from different MRI scans, we need to resample them to the same resolution (0.2mm isotropic in this case) and to the same space. 
To reduce the computational load, we here crop the mri scan (qT1, or VDM for us) to the size of the ROI mask and only resample that. 
This of course assumes that the original ROI (CA1 mask obtained from ASHS) is already registered to the MRI scan. 
After cropping the MRI to the size of the ROI, we resample it to our upsampled masks, using nibabels resample_from_to() function. 
"""

# ------------------ PATHS ------------------ #
mri_path = "path/to/subject_Folders"
sides = ["LH", "RH"]

def _load_nifti(filepath):
    img = nib.load(filepath)
    return img, img.affine

subjects = sorted(os.listdir(mri_path))
print(subjects)

def _crop_mri_sides(mri, roi_mask):
    data_mri = mri.get_fdata().copy()
    data_roi = roi_mask.get_fdata().astype(np.uint8)

    # Step 1: zero out voxels outside ROI
    data_mri[data_roi != 1] = 0

    # Step 2: bounding box crop to nonzero region
    nonzero_coords = np.array(np.nonzero(data_mri))
    min_coords = nonzero_coords.min(axis=1)
    max_coords = nonzero_coords.max(axis=1) + 1

    crop = data_mri[
        min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1],
        min_coords[2]:max_coords[2]
    ]

    return crop.astype(np.float32), min_coords

def _upsample_mri_crop(cropped_mri_nii, ups_roi):
    upsampled_img = resample_from_to(cropped_mri_nii, ups_roi)
    return upsampled_img


# ------------------ MAIN LOOP ------------------ #
all_rows = []

for sub in subjects:
    if sub == ".DS_Store": # Some invisible file that kept appearing in the subject folders directory...
        continue
    print(f"\nProcessing {sub} ...")
    sub_dir = os.path.join(mri_path, sub)
    qt1_path = os.path.join(mri_path, sub, f"{sub}_T1map.nii")
    qt1_img, qt1_affine = _load_nifti(qt1_path)

    for side in sides:
        roi_path = os.path.join(mri_path, sub, f"r{sub}_mask_CA1_{side}.nii")
        roi_ups_path = os.path.join(mri_path, sub, "upsampled_data", f"{sub}_CA1_{side}_HB_upsampled_mask.nii.gz")

        if not os.path.exists(roi_path):
            print(f"  Missing CA1 layer mask for {side}, skipping.")
            continue

        roi_ups, _ = _load_nifti(roi_ups_path)
        roi, roi_affine = _load_nifti(roi_path)

        crop_mri, min_coords = _crop_mri_sides(qt1_img, roi)

        # Update affine to reflect new bounding box origin
        new_affine = qt1_affine.copy()
        new_affine[:3, 3] = qt1_affine[:3, 3] + qt1_affine[:3, :3] @ min_coords

        crop_mri_nii = Nifti1Image(crop_mri, new_affine)
        ups_mri = _upsample_mri_crop(crop_mri_nii, roi_ups)

        output_path = os.path.join(mri_path, sub, "upsampled_data", f"{sub}_qt1_{side}_cropped_upsampled.nii.gz")
        nib.save(ups_mri, output_path)
        print(f"Finished Side {side} \nfor Subject \n{sub}.\nSaved at: {output_path}")

print("All Subjects Finished!"


