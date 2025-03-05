import os
import numpy as np
import nibabel as nib
from scipy.spatial import KDTree
from tqdm import tqdm

def load_fracture_labels(label_dir):
    """
    Loads ground truth fracture label NIFTI images from a directory,
    calculates the mean coordinates for each fracture, and stores them in a dictionary.
    
    Args:
        label_dir (str): Path to the directory containing NIFTI label files.
    
    Returns:
        dict: Dictionary structured as {ribcage_name: [{"coordinates": (x, y, z), "fracture_type": ""}, ...]}
    """
    ribcage_fractures = {}

    # Get list of NIFTI files
    nifti_files = [f for f in os.listdir(label_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]

    # Progress bar loop
    for filename in tqdm(nifti_files, desc="Processing NIFTI Label Files", unit="file"):
        file_path = os.path.join(label_dir, filename)
        ribcage_name = os.path.splitext(filename)[0]  # Remove extension

        # Load the NIFTI label file
        label_nifti = nib.load(file_path)
        label_data = label_nifti.get_fdata()  # Convert to NumPy array
        
        fractures = []
        
        # Identify unique labels in the segmentation mask (excluding background = 0)
        unique_labels = np.unique(label_data)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background (0)

        for label in unique_labels:
            # Get voxel coordinates where this label is present
            coords = np.argwhere(label_data == label)  # Shape: (N, 3)

            if coords.size > 0:
                # Compute mean coordinates (centroid) of this fracture
                mean_coord = np.mean(coords, axis=0)
                mean_coord_tuple = tuple(mean_coord.tolist())  # Convert to tuple

                # Store the fracture data
                fractures.append({"coordinates": mean_coord_tuple, "fracture_type": ""})
        
        # Store in dictionary
        ribcage_fractures[ribcage_name] = fractures
    
    return ribcage_fractures

def evaluate_fracture_detection(gt_fracture_points, pred_fracture_points, threshold=5.0):
    """Computes metrics between ground truth and predicted fracture locations in a point cloud. """
    if len(gt_fracture_points) == 0:
        print("Warning: No ground truth fractures available!")
        return 0, len(pred_fracture_points), 0
    
    if len(pred_fracture_points) == 0:
        print("Warning: No predicted fractures available!")
        return 0, 0, len(gt_fracture_points)

    # Build KDTree for ground truth fracture locations, we can then find the nearest point to a set of coordinates
    gt_tree = KDTree(gt_fracture_points)

    # Check which predicted fractures match ground truth
    distances, indices = gt_tree.query(pred_fracture_points, distance_upper_bound=threshold)

    TP = np.sum(distances < threshold)
    FP = len(pred_fracture_points) - TP
    FN = len(gt_fracture_points) - len(set(indices[distances < threshold]))

    return TP, FP, FN

# Insert code to load segmentation voxels
# Insert code to align voxels field of view with point cloud field of view and convert to x, y, z coordinates
# Insert code to create arrays / dictionaries for the ground truth CSV data and
# do the same for the fracture detection from the skeletons (as shown in the example below)

"""
ribcage_fractures = {
    "Ribcage_001": [
        {"coordinates": (10.5, 20.3, 30.7), "fracture_type": "Displaced"},
        {"coordinates": (15.2, 25.8, 35.1), "fracture_type": "Segmental"},
    ],
    "Ribcage_002": [
        {"coordinates": (40.1, 50.2, 60.3), "fracture_type": "Non Displaced"},
        {"coordinates": (45.7, 55.4, 65.9), "fracture_type": "Displaced"},
        {"coordinates": (48.3, 58.2, 68.6), "fracture_type": "Segmental"},
    ],
    "Ribcage_003": [
        {"coordinates": (70.9, 80.5, 90.2), "fracture_type": "Displaced"},
    ]
}
"""

# Potentially insert classification metrics (segmental, displaced, non-displaced, undetermined)

# TODO: change label_dir to argument parser
label_dir = "/media/DATA_18_TB_2/manolis/ct-shape-analysis/data/ribfrac-train-images-1/Part1_labels"  
ribcage_fracture_data = load_fracture_labels(label_dir)

# Uncomment for sanity check
for ribcage, fractures in ribcage_fracture_data.items():
    print(f"{ribcage}: {fractures}")


# Test arrays
gt_fractures = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])  
pred_fractures = np.array([[10, 21, 29], [39, 49, 61], [100, 100, 100]])  

TP, FP, FN = evaluate_fracture_detection(gt_fractures, pred_fractures, threshold=5.0)
print(f"True Positives: {TP}, False Positives: {FP}, False Negatives: {FN}")
