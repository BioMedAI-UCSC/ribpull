import numpy as np
from scipy.ndimage import distance_transform_edt
import argparse
import os
from skimage import measure
import trimesh

def create_volume_from_labeled_points(points, labels, volume_shape):
    """
    Convert labeled point cloud to binary volume.
    
    Args:
        points: Nx3 array of coordinates
        labels: N-length array with 1 for foreground, 0 for background
        volume_shape: Tuple of (depth, height, width) for volume size
    
    Returns:
        Binary numpy array of given shape
    """
    # Initialize empty volume
    volume = np.zeros(volume_shape, dtype=bool)
    
    # Get only foreground points
    foreground_points = points[labels == 1].astype(int)
    
    # Ensure all points are within bounds
    valid_mask = (
        (foreground_points[:, 0] >= 0) & (foreground_points[:, 0] < volume_shape[0]) &
        (foreground_points[:, 1] >= 0) & (foreground_points[:, 1] < volume_shape[1]) &
        (foreground_points[:, 2] >= 0) & (foreground_points[:, 2] < volume_shape[2])
    )
    foreground_points = foreground_points[valid_mask]
    
    # Set voxels to True where foreground points exist
    if len(foreground_points) > 0:
        volume[foreground_points[:, 0], foreground_points[:, 1], foreground_points[:, 2]] = True
    
    return volume

def compute_sdf(binary_volume):
    """
    Compute approximate signed distance field from binary volume.
    
    Args:
        binary_volume: Boolean numpy array
    
    Returns:
        Float numpy array of same shape with approximate signed distances
    """
    # Compute distance from surface to each outside point
    outside_distance = distance_transform_edt(~binary_volume)
    
    # Compute distance from surface to each inside point
    inside_distance = distance_transform_edt(binary_volume)
    
    # Combine to get signed distance field
    # Points outside have positive distances
    # Points inside have negative distances
    sdf = outside_distance - inside_distance
    
    return sdf

def process_labeled_ct_scan(points, labels, volume_shape=None):
    """
    Complete pipeline to convert labeled CT scan points to SDF.
    
    Args:
        points: Nx3 array of scan coordinates
        labels: N-length array with 1 for foreground, 0 for background
        volume_shape: Optional. If None, will be determined from points.
    
    Returns:
        SDF as numpy array and binary volume
    """
    # Determine volume shape if not provided
    if volume_shape is None:
        max_coords = np.max(points, axis=0)
        volume_shape = tuple(max_coords.astype(int) + 1)
    
    # Create binary volume
    binary_volume = create_volume_from_labeled_points(points, labels, volume_shape)
    
    # Compute SDF
    sdf = compute_sdf(binary_volume)
    
    return sdf, binary_volume

def extract_isosurface(sdf, level=0.0):
    """
    Extract isosurface from SDF using marching cubes.
    
    Args:
        sdf: 3D numpy array with SDF values
        level: Isosurface level (0.0 for surface)
        
    Returns:
        vertices, faces from marching cubes
    """
    # Extract the isosurface using marching cubes
    verts, faces, normals, _ = measure.marching_cubes(sdf, level=level)
    
    return verts, faces, normals

def save_isosurface_as_ply(verts, faces, normals, output_path):
    """
    Save isosurface as PLY file.
    
    Args:
        verts: Vertices from marching cubes
        faces: Faces from marching cubes
        normals: Vertex normals
        output_path: Path to save PLY file
    """
    # Create mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # Save as PLY
    mesh.export(output_path)
    
    return True

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert CT point cloud to SDF')
    parser.add_argument('points_file', type=str, help='Path to points numpy file (.npy)')
    parser.add_argument('labels_file', type=str, help='Path to labels numpy file (.npy)')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save output files')
    parser.add_argument('--output_prefix', type=str, default='output', help='Prefix for output files')
    parser.add_argument('--volume_shape', type=int, nargs=3, help='Custom volume shape (depth height width)')
    parser.add_argument('--save_isosurface', action='store_true', help='Save isosurface as PLY mesh')
    
    args = parser.parse_args()
    
    # Load input files
    print(f"Loading points from {args.points_file}")
    points = np.load(args.points_file)
    
    print(f"Loading labels from {args.labels_file}")
    labels = np.load(args.labels_file)
    
    # Set volume shape if provided
    volume_shape = tuple(args.volume_shape) if args.volume_shape else None
    
    # Process the scan
    print("Computing SDF...")
    sdf, binary_volume = process_labeled_ct_scan(points, labels, volume_shape)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    sdf_output_path = os.path.join(args.output_dir, f"{args.output_prefix}_sdf.npy")
    volume_output_path = os.path.join(args.output_dir, f"{args.output_prefix}_volume.npy")
    
    print(f"Saving SDF to {sdf_output_path}")
    np.save(sdf_output_path, sdf)
    
    print(f"Saving binary volume to {volume_output_path}")
    np.save(volume_output_path, binary_volume)
    
    # Optionally extract and save isosurface
    if args.save_isosurface:
        isosurface_path = os.path.join(args.output_dir, f"{args.output_prefix}_isosurface.ply")
        print(f"Extracting isosurface...")
        verts, faces, normals = extract_isosurface(sdf)
        
        print(f"Saving isosurface to {isosurface_path}")
        save_isosurface_as_ply(verts, faces, normals, isosurface_path)
    
    print("Processing complete!")
    print(f"SDF shape: {sdf.shape}")
    print(f"Min SDF value: {np.min(sdf)}, Max SDF value: {np.max(sdf)}")

if __name__ == "__main__":
    main()