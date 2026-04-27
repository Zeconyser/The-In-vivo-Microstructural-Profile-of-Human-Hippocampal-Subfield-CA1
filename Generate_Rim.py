import os
import numpy as np
import nibabel as nib

''' 
In order for LayNii to generate equidistant layers, the ROI masks need a three-label structure. LayNii generates layers from voxels with label 1, - to voxels with label 2. Ideally then, we want a one-voxel thick outline to our mask, where surfaces have these labels. 
The internal "body" of the mask should have label 3. 
This script is specific to CA1's geometry. However the pass-logic can easily be adjusted to fit other ROIs as well.                                 
'''

# ---------- Settings ----------
input_dir = "path/to/subject_folders"
output_dir = "path/to/subject_folders"
sides = ["LH", "RH"]

os.makedirs(output_dir, exist_ok=True)

# ---------- Pass function ----------
def rim_pass(volume, mask, pass_type, label):
    """
    volume: output volume (uint8)
    mask: CA1 mask (bool)
    pass_type: "lat_med", "med_lat", "inf_sup", "sup_inf"
    label: 1 (inner) or 2 (outer)

    Only assigns label if voxel is currently 0
    """
    S = mask.shape[2]

    for z in range(S):
        slice_mask = mask[:, :, z]

        # --- Column-based passes ---
        if pass_type == "med_lat":
            for col in range(slice_mask.shape[1]):
                rows = np.where(slice_mask[:, col])[0]
                if rows.size > 0:
                    r = rows[0]
                    if volume[r, col, z] == 0:
                        volume[r, col, z] = label

        elif pass_type == "lat_med":
            for col in reversed(range(slice_mask.shape[1])):
                rows = np.where(slice_mask[:, col])[0]
                if rows.size > 0:
                    r = rows[-1]
                    if volume[r, col, z] == 0:
                        volume[r, col, z] = label

        # --- Row-based passes ---
        elif pass_type == "inf_sup":
            for row in range(slice_mask.shape[0]):
                cols = np.where(slice_mask[row, :])[0]
                if cols.size > 0:
                    c = cols[0]
                    if volume[row, c, z] == 0:
                        volume[row, c, z] = label

        elif pass_type == "sup_inf":
            for row in reversed(range(slice_mask.shape[0])):
                cols = np.where(slice_mask[row, :])[0]
                if cols.size > 0:
                    c = cols[-1]
                    if volume[row, c, z] == 0:
                        volume[row, c, z] = label

        else:
            raise ValueError("Unknown pass_type")

    return volume


# ---------- Main loop ----------
subjects = [s for s in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, s))]

for subject in sorted(subjects):
    for side in sides:

        in_file = os.path.join(
            input_dir, subject, "upsampled_data", f"{subject}_CA1_{side}_HB_upsampled_mask.nii.gz"
        )

        if not os.path.exists(in_file):
            print(f"Missing file: {in_file}")
            continue

        print(f"Processing {subject} {side}...")

        # ---------- Load ----------
        nii = nib.load(in_file)
        mask = nii.get_fdata().astype(np.uint8)
        affine = nii.affine
        header = nii.header

        # ---------- Initialize output ----------
        combined = np.zeros_like(mask, dtype=np.uint8)

        is_right = (side == "RH")

        # ---------- PASS ORDER ----------

        if not is_right:
            # ===== LEFT CA1 =====

            # Outer rim
            combined = rim_pass(combined, mask, "med_lat", 2)
            combined = rim_pass(combined, mask, "inf_sup", 2)

            # Inner rim
            combined = rim_pass(combined, mask, "lat_med", 1)
            combined = rim_pass(combined, mask, "sup_inf", 1)

        else:
            # ===== RIGHT CA1 (mirrored) =====

            # Outer rim
            combined = rim_pass(combined, mask, "med_lat", 2)
            combined = rim_pass(combined, mask, "lat_med", 1)

            # Inner rim
            combined = rim_pass(combined, mask, "inf_sup", 1)
            combined = rim_pass(combined, mask, "sup_inf", 2)

        # ---------- Fill body ----------
        combined[np.logical_and(mask, combined == 0)] = 3

        # ---------- Save ----------
        out_path = os.path.join(
            output_dir, subject, "upsampled_data", f"{subject}_CA1_{side}_upsampled_rim.nii.gz"
        )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        nib.save(nib.Nifti1Image(combined, affine, header), out_path)

        print(f"Saved: {out_path}")
        print(f"Labels: {np.unique(combined)}")

print("Done!")


