#!/bin/bash

'''
This script utilizes the LN2_Layers function by LayNii, to generate equidistant Layers between the borders of our ROI mask.
'''

# Base directories
RIM_BASE="path/to/subject_Folders"

# Path to LN2_LAYERS
LN2_LAYERS_BIN=~/laynii/LN2_LAYERS

# Number of layers
NR_LAYERS=21

# Loop over all subjects in Rim_Files
for subj_dir in "${RIM_BASE}"/sub-*; do
    subj_id=$(basename "${subj_dir}")
    echo "Processing subject: ${subj_id}"

    # Loop over both hemispheres
    for side in LH RH; do
        rim_file="${subj_dir}/upsampled_data/${subj_id}_CA1_${side}_upsampled_rim.nii.gz"
        
        layer_dir="${subj_dir}/${subj_id}/upsampled_layer_files"
        mkdir -p "${layer_dir}"

        if [[ -f "${rim_file}" ]]; then
            echo "  Running LN2_LAYERS for ${side}..."
            "${LN2_LAYERS_BIN}" \
                -rim "${rim_file}" \
                -nr_layers "${NR_LAYERS}" \
                -output "${layer_dir}/${subj_id}_${side}_CA1.nii.gz"
        else
            echo "  WARNING: Rim file not found for ${side}: ${rim_file}"
        fi
    done

    echo "Finished subject: ${subj_id}"
    echo "------------------------------------"
done

echo "All subjects processed!"
