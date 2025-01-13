import nibabel as nib
import numpy as np
from pathlib import Path

def analyze_intensity_range(nifti_path):
    """
    Analyze the intensity distribution of a NIfTI file.
    Note: Normally Hounsfield Unit for bone is above 300, but I didn't get many points so I just set it to get a percentage of highest values
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    return {
        'min': np.min(data),
        'max': np.max(data),
        'mean': np.mean(data),
        'median': np.median(data),
        'percentiles': {
            '1%': np.percentile(data, 1),
            '5%': np.percentile(data, 5),
            '25%': np.percentile(data, 25),
            '75%': np.percentile(data, 75),
            '95%': np.percentile(data, 95),
            '99%': np.percentile(data, 99)
        }
    }

def nifti_to_pointcloud(nifti_path, threshold=0, spacing=None):
    """
    Convert a NIfTI file to a point cloud.
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    if spacing is None:
        spacing = img.header.get_zooms()[:3]
    
    binary_data = data > threshold
    points = np.array(np.where(binary_data)).T
    points = points * spacing
    
    return points

def save_pointcloud_obj(points, output_path):
    """
    Save point cloud to OBJ file.
    """
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save as OBJ format
    with open(output_path, 'w') as f:
        # Write vertices
        for point in points:
            f.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        
        # Add points as geometry (optional but helps some viewers)
        for i in range(1, len(points) + 1):
            f.write(f"p {i}\n")

def process_nifti_files(ct_path, seg_path, output_dir, ct_threshold=None):
    """
    Process both CT and segmentation NIfTI files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze intensity ranges
    ct_stats = analyze_intensity_range(ct_path)
    
    # Workaround to get a percentage of highest values (TO BE FIXED)
    if ct_threshold is None:
        ct_threshold = ct_stats['percentiles']['95%']
    
    # Process CT data
    ct_points = nifti_to_pointcloud(ct_path, threshold=ct_threshold)
    if len(ct_points) > 0:
        save_pointcloud_obj(ct_points, output_dir / 'ct_pointcloud.obj')
        print(f"Classifier point cloud saved with {len(ct_points)} points")
    else:
        print(f"Warning: No points found in CT data above threshold {ct_threshold}")
    
    # Process segmentation data
    seg_points = nifti_to_pointcloud(seg_path, threshold=0.5)
    save_pointcloud_obj(seg_points, output_dir / 'seg_pointcloud.obj')
    print(f"Segmentation point cloud saved with {len(seg_points)} points")
    
    # Save statistics
    with open(output_dir / 'statistics.txt', 'w') as f:
        f.write(f"CT point cloud: {len(ct_points)} points\n")
        f.write(f"Segmentation point cloud: {len(seg_points)} points\n")
        
        if len(ct_points) > 0:
            ct_min = ct_points.min(axis=0)
            ct_max = ct_points.max(axis=0)
            f.write("\nCT Bounding Box:\n")
            f.write(f"Min: {ct_min}\n")
            f.write(f"Max: {ct_max}\n")
        
        if len(seg_points) > 0:
            seg_min = seg_points.min(axis=0)
            seg_max = seg_points.max(axis=0)
            f.write("\nSegmentation Bounding Box:\n")
            f.write(f"Min: {seg_min}\n")
            f.write(f"Max: {seg_max}\n")

if __name__ == "__main__":
    # Replace with your file paths
    ct_path = "RibFrac1-rib-cl.nii"
    seg_path = "RibFrac1-rib-seg.nii"
    output_dir = "pointcloud_output"
    
    # Process the files
    process_nifti_files(ct_path, seg_path, output_dir)
    