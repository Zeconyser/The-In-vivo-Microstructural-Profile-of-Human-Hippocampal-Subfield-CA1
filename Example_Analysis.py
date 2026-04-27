import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --- Paths ---
data_path = "/path/to/subject_folders"

# --- Subjects ---
subjects = os.listdir(data_path)

sides = ["LH", "RH"]
ROI = "CA1"
all_data = []
need_csv = 0   # Set this to 1, if you need to get layer data in a csv file
do_plot = 1

if need_csv:
    # --- Processing Loop ---
    for sub in subjects:
        for side in sides:
            # File paths
            subj_layers_path = os.path.join(data_path, sub, "upsampled_layer_files", f"{sub}_{side}_CA1_layers_equidist.nii.gz")
            subj_qt1_path = os.path.join(data_path, sub, "upsampled_data", f"{sub}_qt1_{side}_cropped_upsampled.nii.gz")

            if not os.path.exists(subj_qt1_path) or not os.path.exists(subj_layers_path):
                print(f"Skipping {sub} | {side} due to missing files.")
                continue

            # Load images
            qt1_img = nib.load(subj_qt1_path)
            layer_img = nib.load(subj_layers_path)

            # --- Load data arrays ---
            qt1_data = qt1_img.get_fdata()
            layer_data = layer_img.get_fdata()

            # Bounding box for non-zero mask
            nonzero_coords = np.array(np.nonzero(layer_data))
            min_coords = nonzero_coords.min(axis=1)
            max_coords = nonzero_coords.max(axis=1) + 1

            layer_crop = layer_data[min_coords[0]:max_coords[0],
                                    min_coords[1]:max_coords[1],
                                    min_coords[2]:max_coords[2]]
            qt1_crop = qt1_data[min_coords[0]:max_coords[0],
                                min_coords[1]:max_coords[1],
                                min_coords[2]:max_coords[2]]

            print(f"[{sub} | {side}] Cropped shapes - Layer mask: {layer_crop.shape}, T1 map: {qt1_crop.shape}")

            # Extract values
            layer_ids = np.unique(layer_crop)
            layer_ids = layer_ids[layer_ids != 0]

            for layer_id in layer_ids:
                mask = layer_crop == layer_id
                qt1_values = qt1_crop[mask]

                if qt1_values.size > 0:
                    mean_qt1 = np.mean(qt1_values)
                    all_data.append({
                        "Subject": sub,
                        "Hemisphere": side.capitalize(),
                        "Layer": int(layer_id),
                        "Myelin": mean_qt1
                    })



    # --- Create DataFrame ---
    df = pd.DataFrame(all_data)
    df["Subject"] = df["Subject"].astype(str)

    # --- Save CSV ---
    df.to_csv("path/to/layer_data.csv", index=False)


# --- Load the CSV with Layer Myelin Data ---
path_final_data = ("path/to/layer_data.csv")
df = pd.read_csv(path_final_data)

if do_plot:
    # --- Flip layer numbering --- Only necessary if you accidentaly assigne labels 1 and 2 the the opposite surfaces during rim generation...if you did it correctly, ignore this. 
    print(df["Layer"])
    df["Layer_flipped"] = 22 - df["Layer"]   
    # --- Assign Layer Zones ---
    def assign_zone(layer):
        if 1 <= layer <= 7:
            return "Inner"
        elif 8 <= layer <= 14:
            return "Middle"
        elif 15 <= layer <= 21:
            return "Outer"
        return None

    df["LayerZone"] = df["Layer_flipped"].apply(assign_zone)
    df = df.dropna(subset=["LayerZone", "Myelin"])

    # --- Summary Table ---
    def get_summary_table(df):
        summary = df.groupby("LayerZone")["Myelin"].agg(["mean", "std", "count"]).reset_index()
        summary = summary.rename(columns={"mean": "Mean_Myelin", "std": "STD_Myelin", "count": "N"})
        summary["LayerZone"] = pd.Categorical(summary["LayerZone"], categories=["Inner", "Middle", "Outer"], ordered=True)
        summary = summary.sort_values("LayerZone")
        summary = summary.rename(columns = {"LayerZone" : "Compartment"})
        print("\nSummary:")
        print(summary)

    get_summary_table(df)

    # --- Plot qT1 Profiles (Layer on x-axis) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True, sharex=True)

    # Colors for zones (Inner=red, Middle=yellow, Outer=blue)
    layer_zones = {"Inner": (1, 7.5), "Middle": (7.5, 14.5), "Outer": (14.5, 21)}
    colors = {"Inner": "#ffcccc", "Middle": "#ffff99", "Outer": "#add8e6"}

    for ax, hemi in zip(axes, ["Rh", "Lh"]):
        hemi_df = df[df["Hemisphere"] == hemi]

        # Plot individual subjects
        for subject in hemi_df["Subject"].unique():
            subj_df = hemi_df[hemi_df["Subject"] == subject]
            ax.plot(subj_df["Layer_flipped"], subj_df["Myelin"], color='black', alpha=0.5)

        # Plot colored zones
        for zone, (start, end) in layer_zones.items():
            ax.axvspan(start, end, color=colors[zone], alpha=0.5, zorder=0)

        # Plot group mean
        group_mean = hemi_df.groupby("Layer_flipped")["Myelin"].mean()
        ax.plot(group_mean.index, group_mean.values, color='red', lw=2, label='Group Mean')
        ax.tick_params(axis='x', labelsize=18)  # Change x tick label font size
        ax.tick_params(axis='y', labelsize=18)  # Change y tick label font size

        ax.set_title(f"qT1 Profile - {hemi} CA1", fontweight='bold', size=24)
        ax.set_xlabel("Layer", size = 18)
        ax.set_ylabel("qT1 (ms)", size= 18)

        ax.set_xlim(2, 21)
        ax.set_ylim(1200, 2800)
        #ax.grid(True)

        # Show only whole numbers on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.yticks(size=18)
    plt.xticks(size=18)
    plt.tight_layout()
    plt.savefig("path/to/where_you_want_to_save_it", bbox_inches = 'tight')
    plt.show()


    plt.tight_layout()
    plt.show()
