import os
import numpy as np
from sklearn.cluster import DBSCAN
import argparse

def load_xyz_file(filepath):
    data = np.loadtxt(filepath)
    return data  # Shape: (N, 6) -> X, Y, Z, Nx, Ny, Nz

def save_xyz_file(filepath, data):
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
    """Process all XYZ files in input_dir and save cleaned files in output_dir."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.endswith(".xyz")]

    for file in files:
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)

        print(f"Processing {file} ...")
        data = load_xyz_file(input_path)
        cleaned_data = denoise_point_cloud(data, eps, min_samples)
        save_xyz_file(output_path, cleaned_data)
        print(f"Saved cleaned file: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise XYZ point clouds using DBSCAN clustering.")
    parser.add_argument("input_dir", type=str, help="Path to the directory with input XYZ files.")
    parser.add_argument("output_dir", type=str, help="Path to save denoised XYZ files.")
    parser.add_argument("--eps", type=float, default=0.03, help="DBSCAN epsilon value (distance threshold).")
    parser.add_argument("--min_samples", type=int, default=20, help="DBSCAN minimum samples per cluster.")

    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir, eps=args.eps, min_samples=args.min_samples)
