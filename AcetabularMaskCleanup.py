#-----------------------------------------------------
# AcetabularMaskCleanup.py
#
# Description: Performs hip bone segmentation from input CT images using TotalSegmentator, followed by refinement of acetabular mask
#
# Requirement: Python 3.9 or later; PyTorch 2.0.0 or later
#
# Usage: python AcetabularMaskCleanup.py <input_image> <out_dir>
#
# Accepted file format: NIfTI (.nii/.nii.gz)
#
# Output files:
#   hip_left.nii.gz           – TotalSegmentator output (label map; value = 77)
#   hip_right.nii.gz          – TotalSegmentator output (label map; value = 78)
#   hip_left_cleaned.nii.gz   – refined hip mask (binary)
#   hip_right_cleaned.nii.gz  – refined hip mask (binary)
#-----------------------------------------------------

import os
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import cv2
from totalsegmentator.python_api import totalsegmentator
from skimage import morphology
from skimage.measure import regionprops
from scipy.ndimage import label, generate_binary_structure

# Description: check if spacing is isotropic at a specified target value
# Expected input: 
#   spacing = iterable of floats; (x, y, z) in mm
#   target  = float; target isotropic spacing in mm
#   tol     = float; tolerated error margin
# Output: boolean
def check_spacing(spacing, target=1.5, tol=1e-2):
    return all(abs(s - target) <= tol for s in spacing)

# Description: resample a 3D image to isotropic voxel spacing
# Expected input:
#   image          = SimpleITK.Image; input 3D image
#   target_spacing = float; desired isotropic spacing in mm
#   interpolator   = SimpleITK interpolator mode constant
# Output: SimpleITK.Image
def resample_to_isotropic(image, target_spacing=1.5, interpolator=sitk.sitkLinear):
    # Resample all images to 1.5 mm isotropic resolution
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_spacing = (target_spacing, target_spacing, target_spacing)
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetInterpolator(interpolator)
    return resample.Execute(image)

# Internal helper function for binary thresholding
def _binary_threshold_np(img, low, high):
    arr = np.asarray(img)
    out = np.zeros(arr.shape, dtype=np.uint8)
    m = np.isfinite(arr)
    out[m] = ((arr[m] >= low) & (arr[m] <= high)).astype(np.uint8)
    return out

# Internal helper function for pruning
def _remove_small_objects_np(mask, min_size):
    lab = morphology.label(mask)
    cleaned = morphology.remove_small_objects(lab, min_size)
    return (cleaned > 0).astype(np.uint8)

# Internal helper function for flood-filling
def _floodfill_np(mask):
    arr = np.asarray(mask, dtype=np.uint8)
    h, w = arr.shape
    ff = arr.copy()
    m = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(ff, m, (0,0), 255)
    inv = (cv2.bitwise_not(ff) > 0).astype(np.uint8)
    return inv

# Internal helper function for dilation
def _dilate_np(mask, kernel, it):
    m = np.asarray(mask, dtype=np.uint8)
    k = np.asarray(kernel, dtype=np.uint8)
    i = int(np.asarray(it, dtype=np.uint8))
    return cv2.dilate(m, k, iterations=i)

# Internal helper function for erosion
def _erode_np(mask, kernel, it):
    m = np.asarray(mask, dtype=np.uint8)
    k = np.asarray(kernel, dtype=np.uint8)
    i = int(np.asarray(it, dtype=np.uint8))
    return cv2.erode(m, k, iterations=i)

# Description: refine hip bone segmentation within the acetabular slab using HU thresholding, morphological operations, and 3D connectivity
# Expected input:
#   ct_arr        = np.ndarray; full CT volume (z, y, x) in HU
#   ct_cropped    = np.ndarray; CT volume masked to the hip region (NaN outside ROI)
#   hip_mask      = np.ndarray; binary hip bone mask (z, y, x)
#   dilate_kernel = np.ndarray or None; structuring element for morphological closing
#   dilate_iters  = int; number of dilation iterations
#   erode_iters   = int; number of erosion iterations
# Output: np.ndarray (binary refined hip mask, z, y, x)
def clean_mask_hip(ct_arr, ct_cropped, hip_mask, dilate_kernel=None, dilate_iters=1, erode_iters=1):
    if dilate_kernel is None:
        dilate_kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)

    zdim = hip_mask.shape[0]
    refined_mask = (hip_mask > 0).astype(np.uint8)

    # Find inferior bound of acetabulum corresponding to 0.6 cm below the upper border of obturator foramen
    # The longest run of slices > 2 objects correspond to obturator foramen
    max_len = 0
    current_len = 0
    end_idx = -1
    for z in range(zdim):
        cleaned_slice = _remove_small_objects_np(hip_mask[z], min_size=100)
        _, count = label(cleaned_slice)
        if count >= 2:
            current_len += 1
            if current_len > max_len:
                max_len = current_len
                end_idx = z
        else:
            current_len = 0

    if max_len > 0:
        start_idx = end_idx - max_len + 1
        print(f"Longest run of slices with ≥2 objects: {max_len} slices (z={start_idx} to z={end_idx})")
        print(f"Most superior slice of obturator foramen: z={end_idx}")
    
        inferior_bound = max(end_idx - 3, 0)
        print(f"Inferior bound: z={inferior_bound}")
    else:
        print("Obturator foramen not detected -- falling back to most inferior non-empty slice.")
        
        inferior_bound = None
        for z in range(zdim - 1, -1, -1):
            cleaned_slice = _remove_small_objects_np(hip_mask[z], min_size=100)
            if np.any(cleaned_slice):
                inferior_bound = z
                break
                
        if inferior_bound is None:
            print("Hip bone mask is empty -- returning original mask.")
            return refined_mask

        print(f"Inferior bound (fallback): z={inferior_bound}")

    # Find superior bound corresponding to 0.9 cm above the acetabular roof
    # The first slice with > 0.9 solidity corresponds to acetabular roof
    superior_bound = None
    for z in range(end_idx + 1, zdim):
        labeled, _ = label(hip_mask[z])
        props = regionprops(labeled)
        if props:
            largest_region = max(props, key=lambda r: r.area)
            solidity = largest_region.solidity
        else:
            solidity = 0.0

        if solidity > 0.9:
            superior_bound = z + 6
            if superior_bound >= zdim:
                superior_bound = zdim - 1
            print(f"Acetabular roof at slice {z}, setting superior bound = {superior_bound}")
            break

    if superior_bound is None:
        superior_bound = inferior_bound + 34
        if superior_bound >= zdim:
            superior_bound = zdim - 1
        print("No slice with solidity > 0.9 found. Forcing range to be 34.")

    # Sanity check: valid cleaning range should be within 4.2-6.0 cm
    cleaning_range = superior_bound - inferior_bound
    if cleaning_range < 28 or cleaning_range > 40:
        print(f"Warning: suspicious acetabular roof identification with {cleaning_range} slices of acetabulum. Forcing range to be 34.")
        superior_bound = inferior_bound + 34
        if superior_bound >= zdim:
            superior_bound = zdim - 1

    print(f"Cleaning range: z={inferior_bound} to z={superior_bound}")
    
    # Clean selected slices 
    for i in range(inferior_bound, min(superior_bound + 1, zdim)):
        slice_crop = ct_cropped[i]
        slice_ct = ct_arr[i]

        # Apply binary thresholding
        mask_bi = _binary_threshold_np(slice_crop, low=150, high=np.inf)

        # HU-based expansion
        max_added_voxels = 40
        high_hu = slice_ct > 300
        labeled_high, num_high = label(high_hu.astype(np.uint8))
    
        if num_high > 0:
            touching_labels = np.unique(labeled_high[mask_bi.astype(bool)])
            touching_labels = touching_labels[touching_labels != 0]
    
            if touching_labels.size > 0:
                connected_high = np.isin(labeled_high, touching_labels)
                expanded_mask = mask_bi.astype(bool) | connected_high
                added_voxels = (np.count_nonzero(expanded_mask) - np.count_nonzero(mask_bi))
                
                if added_voxels <= max_added_voxels:
                    mask_exp = expanded_mask.astype(np.uint8)
                else:
                    print(
                        f"Slice {i}: HU-based expansion added {added_voxels} voxels "
                        f"(> {max_added_voxels}); reverting to mask after binary thresholding."
                    )
                    mask_exp = mask_bi
            else:
                mask_exp = mask_bi
        else:
            mask_exp = mask_bi
        mask_dil = _dilate_np(mask_exp, dilate_kernel, dilate_iters)
        mask_erod = _erode_np(mask_dil, dilate_kernel, erode_iters)
        mask_fill = _floodfill_np(mask_erod)
        refined_mask[i] = mask_fill
        
    # 3D connectivity: keep only the largest connected component
    structure = generate_binary_structure(3, 1)
    labeled_3d, num = label(refined_mask.astype(bool), structure=structure)
    print(f"3D connected components found: {num}")
    if num == 0:
        print("Warning: refined hip mask is empty after 3D labeling.")
        return refined_mask
    
    counts = np.bincount(labeled_3d.ravel())
    counts[0] = 0
    
    largest_label = counts.argmax()
    refined_mask = (labeled_3d == largest_label).astype(np.uint8)
    return refined_mask

# Description: run the full acetabular mask refinement workflow
# Expected input:
#   input_image = str; path to input CT NIfTI file (.nii or .nii.gz)
#   out_dir     = str; directory for TotalSegmentator output 
# Output: writes TotalSegmentator output and cleaned hip masks to disk
def run_cleaning_workflow(input_image, out_dir):
    # Load NIfTI file and get shape and affine
    # Check the format the input image
    if not (input_image.lower().endswith(".nii") or input_image.lower().endswith(".nii.gz")):
        raise RuntimeError("Accepted file format: NIfTI (.nii/.nii.gz)")
    if not os.path.isfile(input_image):
        raise RuntimeError(f"Input image does not exist: {input_image}")
    os.makedirs(out_dir, exist_ok=True)

    # Check the spacing of input image and resample to 1.5 mm isotropic if needed
    ct_nii = nib.load(input_image)
    shape = ct_nii.shape
    print("Shape:", shape)
    spacing = ct_nii.header.get_zooms()[:3]   
    print("Spacing (x, y, z):", spacing)
    if check_spacing(spacing, target=1.5, tol=1e-2):
        print("Spacing already 1.5 mm isotropic; skipping resample.")
    if not check_spacing(spacing, target=1.5, tol=1e-2):
        print("Resampling to 1.5 mm isotropic...")
        ct_sitk = sitk.ReadImage(input_image)
        ct_resampled = resample_to_isotropic(ct_sitk, target_spacing=1.5, interpolator=sitk.sitkLinear)
        base = os.path.basename(input_image).replace(".nii.gz", "").replace(".nii", "")
        resampled_path = os.path.join(out_dir, f"{base}_resampled_1p5mm.nii.gz")
        sitk.WriteImage(ct_resampled, resampled_path)
        input_image = resampled_path
        ct_nii = nib.load(input_image)
        shape = ct_nii.shape
        spacing = ct_nii.header.get_zooms()[:3]
        print("Resampled shape:", shape)
        print("Resampled spacing (x, y, z):", spacing)  
    affine = ct_nii.affine

    # Convert to NumPy array and mandate orientation
    ct_arr_init = ct_nii.get_fdata()
    init_axcodes = nib.aff2axcodes(affine)
    print("Initial orientation:", init_axcodes)
    target_axcodes = ('L', 'P', 'S')
    ornt_init = nib.orientations.axcodes2ornt(init_axcodes)
    ornt_target = nib.orientations.axcodes2ornt(target_axcodes)
    ornt_transf = nib.orientations.ornt_transform(ornt_init, ornt_target)
    ct_arr_lps = nib.orientations.apply_orientation(ct_arr_init, ornt_transf)
    ct_arr = np.transpose(ct_arr_lps, axes=(2, 1, 0))
    
    # Run TotalSegmentator
    right_hip_pred = totalsegmentator(input_image, out_dir, task="total", roi_subset=["hip_right"],  fast=False, ml=False, device="gpu", skip_saving=False)
    left_hip_pred  = totalsegmentator(input_image, out_dir, task="total", roi_subset=["hip_left"],   fast=False, ml=False, device="gpu", skip_saving=False)
    # Get segmentation arrays
    right_hip_pred_arr_init = right_hip_pred.get_fdata()
    left_hip_pred_arr_init = left_hip_pred.get_fdata()
    # Correct orientation  
    right_hip_pred_arr_lps = nib.orientations.apply_orientation(right_hip_pred_arr_init, ornt_transf)
    left_hip_pred_arr_lps = nib.orientations.apply_orientation(left_hip_pred_arr_init, ornt_transf)
    # Transpose merged mask    
    right_hip_pred_arr_rev = np.transpose(right_hip_pred_arr_lps, axes=(2, 1, 0))
    left_hip_pred_arr_rev = np.transpose(left_hip_pred_arr_lps, axes=(2, 1, 0))
    # Create cropped image
    right_hip_cropped = np.where(right_hip_pred_arr_rev == 78, ct_arr, np.nan)
    left_hip_cropped = np.where(left_hip_pred_arr_rev == 77, ct_arr, np.nan)
    # Create binary mask
    right_hip_pred_arr_rev_bi = (right_hip_pred_arr_rev == 78).astype(np.uint8)
    left_hip_pred_arr_rev_bi  = (left_hip_pred_arr_rev  == 77).astype(np.uint8)
    # Run post-processing function
    print("\n=== Right hip | default morphological closing ===")
    right_hip_mask_cleaned = clean_mask_hip(ct_arr, right_hip_cropped, right_hip_pred_arr_rev_bi, dilate_kernel=None, dilate_iters=1, erode_iters=1)
    print("\n=== Left hip | default morphological closing ===")
    left_hip_mask_cleaned = clean_mask_hip(ct_arr, left_hip_cropped, left_hip_pred_arr_rev_bi, dilate_kernel=None, dilate_iters=1, erode_iters=1)
    print("\n=== Right hip | fallback morphological closing ===")
    right_hip_mask_cleaned_fallback = clean_mask_hip(ct_arr, right_hip_cropped, right_hip_pred_arr_rev_bi, dilate_kernel=None, dilate_iters=3, erode_iters=3)
    print("\n=== Left hip | fallback morphological closing ===")
    left_hip_mask_cleaned_fallback = clean_mask_hip(ct_arr, left_hip_cropped, left_hip_pred_arr_rev_bi, dilate_kernel=None, dilate_iters=3, erode_iters=3)
        
    # Identify slices where voxel count difference > 60 between the two morph close methods
    right_hip_voxel_count = np.count_nonzero(right_hip_mask_cleaned, axis=(1, 2)).astype(np.int32)
    right_hip_voxel_count_fallback = np.count_nonzero(right_hip_mask_cleaned_fallback, axis=(1, 2)).astype(np.int32)
    right_hip_voxel_count_diff = right_hip_voxel_count_fallback - right_hip_voxel_count
    right_hip_slices_to_replace = np.where(right_hip_voxel_count_diff > 60)[0]
    left_hip_voxel_count = np.count_nonzero(left_hip_mask_cleaned, axis=(1, 2)).astype(np.int32)
    left_hip_voxel_count_fallback = np.count_nonzero(left_hip_mask_cleaned_fallback, axis=(1, 2)).astype(np.int32)
    left_hip_voxel_count_diff = left_hip_voxel_count_fallback - left_hip_voxel_count
    left_hip_slices_to_replace = np.where(left_hip_voxel_count_diff > 60)[0]
    
    # Replace those slices to generate the processed mask
    right_hip_mask_cleaned_processed = right_hip_mask_cleaned.copy()
    left_hip_mask_cleaned_processed = left_hip_mask_cleaned.copy()
    right_hip_mask_cleaned_processed[right_hip_slices_to_replace] = \
        right_hip_mask_cleaned_fallback[right_hip_slices_to_replace]
    left_hip_mask_cleaned_processed[left_hip_slices_to_replace] = \
        left_hip_mask_cleaned_fallback[left_hip_slices_to_replace]    
    print(f"Replaced {len(right_hip_slices_to_replace)} slices for the right hip with fallback morphological closing.")
    print(f"Slices replaced: {right_hip_slices_to_replace.tolist()}")
    print(f"Replaced {len(left_hip_slices_to_replace)} slices for the left hip with fallback morphological closing.")
    print(f"Slices replaced: {left_hip_slices_to_replace.tolist()}")
    
    # Identify slices where voxel count difference < -10 or > 200 between the processed mask and the original mask
    right_hip_mask_cleaned_final = right_hip_mask_cleaned_processed.copy()
    left_hip_mask_cleaned_final = left_hip_mask_cleaned_processed.copy()
    right_hip_voxel_count_processed = np.count_nonzero(right_hip_mask_cleaned_processed, axis=(1, 2)).astype(np.int32)
    right_hip_voxel_count_original = np.count_nonzero(right_hip_pred_arr_rev, axis=(1, 2)).astype(np.int32)
    right_hip_voxel_count_diff_final = right_hip_voxel_count_original - right_hip_voxel_count_processed
    right_hip_slices_to_replace_final = np.where((right_hip_voxel_count_diff_final < -10) | (right_hip_voxel_count_diff_final > 200))[0]
    left_hip_voxel_count_processed = np.count_nonzero(left_hip_mask_cleaned_processed, axis=(1, 2)).astype(np.int32)
    left_hip_voxel_count_original = np.count_nonzero(left_hip_pred_arr_rev, axis=(1, 2)).astype(np.int32)
    left_hip_voxel_count_diff_final = left_hip_voxel_count_original - left_hip_voxel_count_processed
    left_hip_slices_to_replace_final = np.where((left_hip_voxel_count_diff_final < -10) | (left_hip_voxel_count_diff_final > 200))[0]

    # Replace those slices to generate the final mask
    right_hip_mask_cleaned_final[right_hip_slices_to_replace_final] = \
        right_hip_pred_arr_rev_bi[right_hip_slices_to_replace_final]
    left_hip_mask_cleaned_final[left_hip_slices_to_replace_final] = \
        left_hip_pred_arr_rev_bi[left_hip_slices_to_replace_final]
    print(f"Replaced {len(right_hip_slices_to_replace_final)} slices for the right hip with original segmentation.")
    print(f"Slices replaced: {right_hip_slices_to_replace_final.tolist()}")
    print(f"Replaced {len(left_hip_slices_to_replace_final)} slices for the left hip with original segmentation.")
    print(f"Slices replaced: {left_hip_slices_to_replace_final.tolist()}")

    # Transpose back
    right_hip_mask_cleaned_transposed = np.transpose(right_hip_mask_cleaned_final, (2, 1, 0))
    left_hip_mask_cleaned_transposed = np.transpose(left_hip_mask_cleaned_final, (2, 1, 0))
    # Reorient back
    ornt_transf_back = nib.orientations.ornt_transform(ornt_target, ornt_init)
    right_hip_mask_cleaned_ras = nib.orientations.apply_orientation(right_hip_mask_cleaned_transposed, ornt_transf_back)
    left_hip_mask_cleaned_ras = nib.orientations.apply_orientation(left_hip_mask_cleaned_transposed, ornt_transf_back)
    # Save cleaned mask as NIfTI file that matches the orientation of original input file
    right_hip_mask_cleaned_nii = nib.Nifti1Image(right_hip_mask_cleaned_ras, affine=ct_nii.affine, header=ct_nii.header)
    left_hip_mask_cleaned_nii = nib.Nifti1Image(left_hip_mask_cleaned_ras, affine=ct_nii.affine, header=ct_nii.header)
    # Save to disk
    nib.save(right_hip_mask_cleaned_nii, os.path.join(out_dir, 'hip_right_cleaned.nii.gz'))
    nib.save(left_hip_mask_cleaned_nii, os.path.join(out_dir, 'hip_left_cleaned.nii.gz'))

# Build the command-line argument parser
def build_parser():
    p = argparse.ArgumentParser(
        description="Runs hip bone segmentation (TotalSegmentator) and acetabular mask refinement."
    )
    p.add_argument("input_image", type=str, help="Input CT NIfTI file (.nii or .nii.gz)")
    p.add_argument("out_dir", type=str, help="Output directory for segmentations")
    return p

# Entry point for command-line execution
def main(argv=None):
    args = build_parser().parse_args(argv)
    run_cleaning_workflow(args.input_image, args.out_dir)

if __name__ == "__main__":
    raise SystemExit(main())

    