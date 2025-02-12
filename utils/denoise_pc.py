import os
import numpy as np
from sklearn.cluster import DBSCAN
import argparse

def normalize_point_cloud(points):
    """
    Normalize point cloud to fit in a unit cube centered at origin.
    
    Parameters:
        points: (N, 3) or (N, 6) array of points (and possibly normals)
    
    Returns:
        normalized points, scale factor, center
    """
    # Extract just the XYZ coordinates for normalization
    xyz = points[:, :3]
    
    # Calculate center and move to origin
    center = np.mean(xyz, axis=0)
    centered = xyz - center
    
    # Scale to unit cube while preserving aspect ratio
    scale = np.max(np.abs(centered))
    normalized = centered / scale
    
    # If we have normals, keep them unchanged
    if points.shape[1] > 3:
        normalized_points = np.hstack([normalized, points[:, 3:]])
    else:
        normalized_points = normalized
        
    return normalized_points, scale, center

def denormalize_point_cloud(points, scale, center):
    """
    Restore the original scale and position of the point cloud.
    """
    xyz = points[:, :3] * scale + center
    if points.shape[1] > 3:
        return np.hstack([xyz, points[:, 3:]])
    return xyz

def load_ply_file(filepath):
    """
    Load PLY file and extract XYZ coordinates and normals if available.
    Returns normalized data in the same format as XYZ files (N, 6) -> X, Y, Z, Nx, Ny, Nz
    """
    vertices = []
    normals = []
    with open(filepath, 'r') as f:
        # Parse header
        n_vertices = 0
        has_normals = False
        header_end = False
        
        while not header_end:
            line = f.readline().strip()
            if line == "end_header":
                header_end = True
            elif "element vertex" in line:
                n_vertices = int(line.split()[-1])
            elif "property float nx" in line:
                has_normals = True
        
        # Read vertex data
        for _ in range(n_vertices):
            line = f.readline().strip().split()
            vertices.append([float(x) for x in line[:3]])  # XYZ
            if has_normals and len(line) >= 6:
                normals.append([float(x) for x in line[3:6]])  # NxNyNz
            else:
                normals.append([0.0, 0.0, 0.0])  # Default normals if not present
    
    # Convert to numpy arrays and normalize
    data = np.hstack([np.array(vertices), np.array(normals)])
    normalized_data, scale, center = normalize_point_cloud(data)
    return normalized_data, scale, center

def save_ply_file(filepath, data, scale=None, center=None):
    """Save point cloud data to PLY file format, denormalizing if scale and center are provided."""
    if scale is not None and center is not None:
        data = denormalize_point_cloud(data, scale, center)
        
    with open(filepath, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(data)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("end_header\n")
        
        # Write vertex data
        for point in data:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                   f"{point[3]:.6f} {point[4]:.6f} {point[5]:.6f}\n")

def load_xyz_file(filepath):
    # XYZ files are assumed to be already normalized
    data = np.loadtxt(filepath)
    return data, None, None  # Return None for scale and center since already normalized

def save_xyz_file(filepath, data, scale=None, center=None):
    # For XYZ files, we keep them normalized
    np.savetxt(filepath, data, fmt="%.6f")

def denoise_point_cloud(data, eps=0.05, min_samples=10):
    """
    Apply DBSCAN clustering to remove noise.
    
    Parameters:
        - data: (N, 6) NumPy array containing XYZ and normals
        - eps: Maximum distance between points in a cluster
        - min_samples: Minimum number of points to form a cluster
    
    Returns:
        - filtered_data: (M, 6) NumPy array of denoised points
    """
    xyz = data[:, :3]  # Extract XYZ coordinates
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
    
    labels = clustering.labels_
    valid_mask = labels != -1  # Keep only points in clusters (ignore noise)
    return data[valid_mask]

def process_directory(input_dir, output_dir, eps=0.05, min_samples=10):
    """Process all XYZ and PLY files in input_dir and save cleaned files in output_dir."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process both XYZ and PLY files
    files = [f for f in os.listdir(input_dir) if f.endswith((".xyz", ".ply"))]
    
    for file in files:
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)
        print(f"Processing {file} ...")
        
        # Load data based on file extension
        if file.endswith(".xyz"):
            data, scale, center = load_xyz_file(input_path)
        else:  # .ply file
            data, scale, center = load_ply_file(input_path)
        
        cleaned_data = denoise_point_cloud(data, eps, min_samples)
        
        # Save data based on file extension
        if file.endswith(".xyz"):
            save_xyz_file(output_path, cleaned_data, scale, center)
        else:  # .ply file
            save_ply_file(output_path, cleaned_data, scale, center)
            
        print(f"Saved cleaned file: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise XYZ and PLY point clouds using DBSCAN clustering.")
    parser.add_argument("input_dir", type=str, help="Path to the directory with input XYZ/PLY files.")
    parser.add_argument("output_dir", type=str, help="Path to save denoised XYZ/PLY files.")
    parser.add_argument("--eps", type=float, default=0.03, help="DBSCAN epsilon value (distance threshold).")
    parser.add_argument("--min_samples", type=int, default=20, help="DBSCAN minimum samples per cluster.")
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir, eps=args.eps, min_samples=args.min_samples)