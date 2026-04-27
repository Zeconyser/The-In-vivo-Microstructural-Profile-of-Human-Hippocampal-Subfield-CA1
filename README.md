# The-In-vivo Microstructural Profile of Human Hippocampal Subfield CA1
This repository contains the Code used for analysis in the manuscript 

"The In-vivo Microstructural Profile of Human Hippocampal Subfield CA1 and its Relation to Memory Performance"

To replicate the pipeline, as described in the manuscript, the files would be run in the following order: 

1. Upsample_binary_ROI_masks.py
2. Generate_Rim.py
3. Generate_Layers.sh
4. Crop_and_Upsample_MRI.py
5. Example_analysis (Layer Plots)

This script assumes that you posess your binary ROI masks in the space of the MRI modality that you wish to analyze. 


