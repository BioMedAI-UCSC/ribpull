import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
import argparse
import os
from skimage import measure
import trimesh
import denoise_pc

def create_volume_from_labeled_points(points, labels, volume_shape, spacing=None):
    """
    Convert labeled point cloud to binary volume.
    
    Args:
        points: Nx3 array of coordinates
        labels: N-length array with 1 for foreground, 0 for background
        volume_shape: Tuple of (depth, height, width) for volume size
        spacing: Optional tuple for CT scan spacing (z, y, x)
    
    Returns:
        Binary numpy array of given shape
    """
    # Initialize empty volume
    volume = np.zeros(volume_shape, dtype=bool)
    
    # Scale points if spacing is provided
    if spacing is not None:
        scaled_points = points.copy()
        for i in range(3):
            scaled_points[:, i] = scaled_points[:, i] / spacing[i]
        points_to_use = scaled_points
    else:
        points_to_use = points
    
    # Get only foreground points
    foreground_points = points_to_use[labels == 1].astype(int)
    
    # Ensure all points are within bounds
    valid_mask = (
        (foreground_points[:, 0] >= 0) & (foreground_points[:, 0] < volume_shape[0]) &
        (foreground_points[:, 1] >= 0) & (foreground_points[:, 1] < volume_shape[1]) &
        (foreground_points[:, 2] >= 0) & (foreground_points[:, 2] < volume_shape[2])
    )
    
    # Count out-of-bounds points
    out_of_bounds = np.sum(~valid_mask)
    if out_of_bounds > 0:
        print(f"Warning: {out_of_bounds} points out of bounds and ignored")
    
    foreground_points = foreground_points[valid_mask]
    
    # Set voxels to True where foreground points exist
    if len(foreground_points) > 0:
        volume[foreground_points[:, 0], foreground_points[:, 1], foreground_points[:, 2]] = True
    else:
        print("Warning: No valid foreground points in volume")
    
    return volume

def compute_sdf(binary_volume, sigma=1.0):
    """
    Compute approximate signed distance field from binary volume.
    
    Args:
        binary_volume: Boolean numpy array
        sigma: Gaussian smoothing sigma (0 for no smoothing)
    
    Returns:
        Float numpy array of same shape with approximate signed distances
    """
    if np.sum(binary_volume) == 0:
        print("Warning: Empty binary volume - SDF calculation may be meaningless")
        return np.ones(binary_volume.shape)
    
    if np.all(binary_volume):
        print("Warning: Full binary volume - SDF calculation may be meaningless")
        return -np.ones(binary_volume.shape)
    
    # Compute distance from surface to each outside point
    outside_distance = distance_transform_edt(~binary_volume)
    
    # Compute distance from surface to each inside point
    inside_distance = distance_transform_edt(binary_volume)
    
    # Combine to get signed distance field
    # Points outside have positive distances
    # Points inside have negative distances
    sdf = outside_distance - inside_distance
    
    # Apply Gaussian smoothing if sigma > 0
    if sigma > 0:
        print(f"Applying Gaussian smoothing with sigma={sigma}")
        sdf = gaussian_filter(sdf, sigma=sigma)
    
    return sdf

def process_labeled_ct_scan(points, labels, volume_shape=None, spacing=None, dbscan_eps=0.05, 
                           dbscan_min_samples=10, smooth_sigma=1.0):
    """
    Complete pipeline to convert labeled CT scan points to SDF.
    
    Args:
        points: Nx3 array of scan coordinates
        labels: N-length array with 1 for foreground, 0 for background
        volume_shape: Optional. If None, will be determined from points.
        spacing: Optional tuple for CT scan spacing (z, y, x)
        dbscan_eps: DBSCAN epsilon parameter for denoising
        dbscan_min_samples: DBSCAN min_samples parameter
        smooth_sigma: Sigma for Gaussian smoothing (0 for no smoothing)
    
    Returns:
        SDF as numpy array and binary volume
    """
    # Apply denoising
    print(f"Denoising point cloud with DBSCAN (eps={dbscan_eps}, min_samples={dbscan_min_samples})...")
    denoised_points, denoised_labels = denoise_pc.denoise_point_cloud(
        points, labels, eps=dbscan_eps, min_samples=dbscan_min_samples
    )
    
    # Determine volume shape if not provided
    if volume_shape is None:
        if spacing is not None:
            # Scale by spacing if provided
            scaled_points = denoised_points.copy()
            for i in range(3):
                scaled_points[:, i] = scaled_points[:, i] / spacing[i]
            max_coords = np.max(scaled_points, axis=0)
        else:
            max_coords = np.max(denoised_points, axis=0)
        
        volume_shape = tuple(max_coords.astype(int) + 1)
        print(f"Automatically determined volume shape: {volume_shape}")
    
    # Create binary volume
    print("Creating binary volume...")
    binary_volume = create_volume_from_labeled_points(
        denoised_points, denoised_labels, volume_shape, spacing
    )
    
    # Compute SDF
    print("Computing SDF...")
    sdf = compute_sdf(binary_volume, sigma=smooth_sigma)
    
    return sdf, binary_volume

def extract_isosurface(sdf, level=0.0, spacing=None, attempt_fix=True):
    """
    Extract isosurface from SDF using marching cubes.
    
    Args:
        sdf: 3D numpy array with SDF values
        level: Isosurface level (0.0 for surface)
        spacing: Optional tuple for CT scan spacing (z, y, x)
        attempt_fix: Try alternative levels if extraction fails
        
    Returns:
        vertices, faces, normals from marching cubes or None if extraction fails
    """
    # Check if level is within SDF range
    sdf_min, sdf_max = np.min(sdf), np.max(sdf)
    if level < sdf_min or level > sdf_max:
        print(f"Warning: Surface level {level} is outside SDF range [{sdf_min}, {sdf_max}]")
        if attempt_fix:
            print("Attempting to find valid isosurface level...")
            # Try a level within the range
            level = max(sdf_min + 0.1, min(sdf_max - 0.1, 0.0))
            print(f"Using adjusted level: {level}")
        else:
            return None, None, None
    
    try:
        # Extract the isosurface using marching cubes
        if spacing is not None:
            verts, faces, normals, _ = measure.marching_cubes(
                sdf, level=level, spacing=spacing, allow_degenerate=False
            )
        else:
            verts, faces, normals, _ = measure.marching_cubes(
                sdf, level=level, allow_degenerate=False
            )
        
        print(f"Extracted isosurface with {len(verts)} vertices and {len(faces)} faces")
        return verts, faces, normals
    
    except Exception as e:
        print(f"Error extracting isosurface: {str(e)}")
        
        if attempt_fix and "Surface level must be" in str(e):
            # Try a different level if the error is about surface level
            new_level = (sdf_min + sdf_max) / 2
            print(f"Trying again with level {new_level}...")
            return extract_isosurface(sdf, level=new_level, spacing=spacing, attempt_fix=False)
        
        return None, None, None

def validate_and_clean_mesh(verts, faces, normals):
    """
    Validate and clean mesh to remove issues.
    
    Args:
        verts: Vertices from marching cubes
        faces: Faces from marching cubes
        normals: Vertex normals
    
    Returns:
        Cleaned vertices, faces, normals
    """
    if verts is None or faces is None:
        return None, None, None
    
    try:
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        
        # Check for issues
        if not mesh.is_watertight:
            print("Warning: Mesh is not watertight")
        
        if not mesh.is_winding_consistent:
            print("Fixing inconsistent face winding...")
            mesh.fix_winding()
        
        # Remove duplicate vertices
        initial_vertices = len(mesh.vertices)
        mesh.merge_vertices(merge_tex=False, merge_norm=False)
        if len(mesh.vertices) < initial_vertices:
            print(f"Removed {initial_vertices - len(mesh.vertices)} duplicate vertices")
        
        # Remove degenerate faces
        initial_faces = len(mesh.faces)
        mesh.remove_degenerate_faces()
        if len(mesh.faces) < initial_faces:
            print(f"Removed {initial_faces - len(mesh.faces)} degenerate faces")
        
        # Fill holes (if needed)
        if not mesh.is_watertight and len(mesh.faces) > 0:
            print("Attempting to fill holes...")
            mesh.fill_holes()
        
        return mesh.vertices, mesh.faces, mesh.vertex_normals
    
    except Exception as e:
        print(f"Error cleaning mesh: {str(e)}")
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
    if verts is None or faces is None:
        print("Error: Cannot save mesh - vertices or faces are missing")
        return False
    
    try:
        # Create mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        
        # Save as PLY
        mesh.export(output_path)
        print(f"Mesh saved successfully to {output_path}")
        return True
    
    except Exception as e:
        print(f"Error saving mesh: {str(e)}")
        return False

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert CT point cloud to SDF')
    parser.add_argument('points_file', type=str, help='Path to points numpy file (.npy)')
    parser.add_argument('labels_file', type=str, help='Path to labels numpy file (.npy)')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save output files')
    parser.add_argument('--output_prefix', type=str, default='output', help='Prefix for output files')
    parser.add_argument('--volume_shape', type=int, nargs=3, help='Custom volume shape (depth height width)')
    parser.add_argument('--spacing', type=float, nargs=3, help='CT scan spacing (z y x)')
    parser.add_argument('--save_isosurface', action='store_true', help='Save isosurface as PLY mesh')
    parser.add_argument('--dbscan_eps', type=float, default=0.05, help='DBSCAN epsilon for denoising')
    parser.add_argument('--dbscan_min_samples', type=int, default=10, help='DBSCAN min samples')
    parser.add_argument('--smooth_sigma', type=float, default=1.0, help='Gaussian smoothing sigma (0 for none)')
    
    args = parser.parse_args()
    
    # Load input files
    print(f"Loading points from {args.points_file}")
    try:
        points = np.load(args.points_file)
        print(f"Loaded {len(points)} points")
    except Exception as e:
        print(f"Error loading points file: {str(e)}")
        return
    
    print(f"Loading labels from {args.labels_file}")
    try:
        labels = np.load(args.labels_file)
        print(f"Loaded {len(labels)} labels")
    except Exception as e:
        print(f"Error loading labels file: {str(e)}")
        return
    
    # Set volume shape if provided
    volume_shape = tuple(args.volume_shape) if args.volume_shape else None
    spacing = tuple(args.spacing) if args.spacing else None
    
    # Process the scan
    print("Processing CT scan...")
    try:
        sdf, binary_volume = process_labeled_ct_scan(
            points, labels, volume_shape, spacing,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            smooth_sigma=args.smooth_sigma
        )
    except Exception as e:
        print(f"Error processing CT scan: {str(e)}")
        return
    
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
        
        verts, faces, normals = extract_isosurface(sdf, spacing=spacing)
        
        # Validate and clean mesh
        verts, faces, normals = validate_and_clean_mesh(verts, faces, normals)
        
        if verts is not None and faces is not None:
            print(f"Saving isosurface to {isosurface_path}")
            save_isosurface_as_ply(verts, faces, normals, isosurface_path)
    
    print("Processing complete!")
    print(f"SDF shape: {sdf.shape}")
    print(f"Min SDF value: {np.min(sdf)}, Max SDF value: {np.max(sdf)}")
    print(f"Binary volume: {np.sum(binary_volume)} foreground voxels out of {binary_volume.size}")

if __name__ == "__main__":
    main()