# Acetabular Segmentation Post-Processing
This repository performs hip bone segmentation from input CT images using [TotalSegmentator](https://github.com/wasserth/TotalSegmentator?tab=readme-ov-file), followed by refinement of the hip bone mask within the acetabulum.

## Repository contents
- `AcetabularMaskCleanup.py`: command-line script to run the full workflow.
- `AcetabularMaskCleanup_Visualization.ipynb`: notebook for visualization of representative cases.
- `AcetabularMaskCleanup_Batch.ipynb`: notebook used in the validation study (link to come) of this post-processing algorithm in the [TotalSegmentator small subset](https://zenodo.org/records/10047263) against [corrected ground truth](https://zenodo.org/records/18853791).
- `totalsegmentatorenv.yml`: conda environment specification.

## Usage
Accepted input format: NIfTI (.nii / .nii.gz)
python AcetabularMaskCleanup.py <input_image> <out_dir>

## Outputs
The script writes the following files to <out_dir>:
- hip_left.nii.gz — TotalSegmentator output (label map; value = 77)
- hip_right.nii.gz — TotalSegmentator output (label map; value = 78)
- hip_left_cleaned.nii.gz — refined hip mask (binary)
- hip_right_cleaned.nii.gz — refined hip mask (binary)
