import numpy as np
from scipy.spatial import KDTree

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

# Test arrays
gt_fractures = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])  
pred_fractures = np.array([[10, 21, 29], [39, 49, 61], [100, 100, 100]])  

TP, FP, FN = evaluate_fracture_detection(gt_fractures, pred_fractures, threshold=5.0)
print(f"True Positives: {TP}, False Positives: {FP}, False Negatives: {FN}")
