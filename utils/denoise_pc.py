import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
import argparse
import os
from skimage import measure
import trimesh
from sklearn.cluster import DBSCAN

def denoise_point_cloud(points, labels, eps=0.05, min_samples=10):
    """
    Apply DBSCAN clustering to remove noise points.
    
    Args:
        points: Nx3 array of scan coordinates
        labels: N-length array with 1 for foreground, 0 for background
        eps: DBSCAN epsilon parameter (distance threshold)
        min_samples: DBSCAN min_samples parameter
    
    Returns:
        Filtered points and labels
    """
    # Only denoise foreground points
    foreground_indices = np.where(labels == 1)[0]
    foreground_points = points[foreground_indices]
    
    # Skip denoising if not enough foreground points
    if len(foreground_points) < min_samples:
        print("Warning: Not enough foreground points for denoising")
        return points, labels
    
    try:
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(foreground_points)
        cluster_labels = clustering.labels_
        
        # Keep only points that belong to clusters (not noise)
        valid_mask = cluster_labels != -1
        
        # Count number of noise points removed
        noise_count = np.sum(~valid_mask)
        if noise_count > 0:
            print(f"Removed {noise_count} noise points ({noise_count / len(foreground_points) * 100:.2f}%)")
        
        # Create new filtered arrays
        filtered_foreground_indices = foreground_indices[valid_mask]
        
        # Create mask for all points
        all_valid_mask = np.ones(len(points), dtype=bool)
        all_valid_mask[foreground_indices[~valid_mask]] = False
        
        return points[all_valid_mask], labels[all_valid_mask]
    
    except Exception as e:
        print(f"Warning: Denoising failed with error: {str(e)}")
        print("Continuing with original point cloud")
        return points, labels
