# Acetabular Segmentation Post-Processing
This repository presents an automated pipeline that performs hip bone segmentation from input CT images using [TotalSegmentator](https://github.com/wasserth/TotalSegmentator?tab=readme-ov-file), followed by refinement of the hip bone mask within the acetabulum.

## Repository contents
- `AcetabularMaskCleanup.py`: command-line script to run the full pipeline.
- `AcetabularMaskCleanup_Visualization.ipynb`: notebook for visualization of representative cases.
- `AcetabularMaskCleanup_Development.ipynb`: notebook used in the batch analysis of this post-processing algorithm within the development dataset [(TotalSegmentator small subset)](https://zenodo.org/records/10047263) against [corrected ground truth](https://zenodo.org/records/18853791).
- `AcetabularMaskCleanup_Validation.ipynb`: notebook used in the batch analysis of this post-processing algorithm within the validation datasets, [KiTS19 subset](https://github.com/neheller/kits19) and [MSD_T10 subset](https://drive.google.com/file/d/1m7tMpE9qEcQGQjL_BdMD-Mvgmc44hG1Y/view), against ground truth from [CTPelvic1K study](https://github.com/MIRACLE-Center/CTPelvic1K).
- `totalsegmentatorenv.yml`: conda environment specification.

## Setup
The workflow requires the following dependencies:
- Python 3.9+
- `numpy`
- `nibabel`
- `SimpleITK`
- `opencv-python`
- `scikit-image` 
- `scipy` 
- `totalsegmentator`
- `torch` 

For users with an NVIDIA GPU and a compatible CUDA setup (PyTorch cu121), the provided conda environment file can be used:

```bash
conda env create -f totalsegmentatorenv.yml
conda activate ts-acetabular-postproc
```

## Usage
Accepted input format: NIfTI (.nii / .nii.gz)

```bash
python AcetabularMaskCleanup.py <input_image> <out_dir>
```

The script writes the following files to <out_dir>:
- hip_left.nii.gz — TotalSegmentator output (single-label mask; value = 77)
- hip_right.nii.gz — TotalSegmentator output (single-label mask; value = 78)
- hip_left_cleaned.nii.gz — refined hip mask (binary)
- hip_right_cleaned.nii.gz — refined hip mask (binary)
