from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_closing, binary_erosion, label
import argparse
import os
from skimage import measure
import trimesh
import numpy as np
import denoise_pc

def create_extrapolated_volume(points, labels, volume_shape, spacing=None, radius=3.0):
    """
    Create a volume by extrapolating from labeled points using distance-weighted influence.
    
    Args:
        points: Nx3 array of coordinates
        labels: N-length array with 1 for foreground, 0 for background
        volume_shape: Tuple of (depth, height, width) for volume size
        spacing: Optional tuple for CT scan spacing (z, y, x)
        radius: Influence radius for each point
    
    Returns:
        Probability volume (float array between 0 and 1)
    """
    # Create empty volume
    volume = np.zeros(volume_shape, dtype=float)
    
    # Scale points if spacing is provided
    if spacing is not None:
        scaled_points = points.copy()
        for i in range(3):
            scaled_points[:, i] = scaled_points[:, i] / spacing[i]
        points_to_use = scaled_points
    else:
        points_to_use = points
    
    # Convert to integer coordinates for voxel grid
    point_coords = np.round(points_to_use).astype(int)
    
    # Ensure all points are within bounds
    valid_mask = (
        (point_coords[:, 0] >= 0) & (point_coords[:, 0] < volume_shape[0]) &
        (point_coords[:, 1] >= 0) & (point_coords[:, 1] < volume_shape[1]) &
        (point_coords[:, 2] >= 0) & (point_coords[:, 2] < volume_shape[2])
    )
    
    point_coords = point_coords[valid_mask]
    point_labels = labels[valid_mask]
    
    # Focus only on foreground points (more efficient)
    foreground_points = point_coords[point_labels == 1]
    
    if len(foreground_points) == 0:
        print("Warning: No valid foreground points in volume")
        return volume
    
    # Process points in batches to improve efficiency
    radius_int = int(radius)
    radius_squared = radius**2
    batch_size = min(1000, len(foreground_points))  # Process 1000 points at a time
    
    for batch_start in range(0, len(foreground_points), batch_size):
        batch_end = min(batch_start + batch_size, len(foreground_points))
        batch_points = foreground_points[batch_start:batch_end]
        
        for point in batch_points:
            # Get bounding box around point (more efficient than processing whole volume)
            z, y, x = point
            z_min, z_max = max(0, z-radius_int), min(volume_shape[0], z+radius_int+1)
            y_min, y_max = max(0, y-radius_int), min(volume_shape[1], y+radius_int+1)
            x_min, x_max = max(0, x-radius_int), min(volume_shape[2], x+radius_int+1)
            
            # Create grid for this subvolume
            z_grid, y_grid, x_grid = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
            
            # Calculate squared distances
            squared_distances = (z_grid-z)**2 + (y_grid-y)**2 + (x_grid-x)**2
            
            # Apply influence based on distance
            influence = np.maximum(0, 1.0 - squared_distances / radius_squared)
            
            # Update the volume - use maximum influence at each voxel
            volume[z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(
                volume[z_min:z_max, y_min:y_max, x_min:x_max], influence
            )
    
    return volume

def remove_small_components(binary_volume, min_size=100):
    """
    Remove small disconnected components from binary volume.
    
    Args:
        binary_volume: Boolean numpy array
        min_size: Minimum size (in voxels) of components to keep
        
    Returns:
        Cleaned binary volume with small components removed
    """
    # Label connected components
    labeled_volume, num_features = label(binary_volume)
    
    if num_features == 0:
        return binary_volume
        
    print(f"Found {num_features} connected components")
    
    # Count voxels in each component
    component_sizes = np.bincount(labeled_volume.ravel())[1:] if num_features > 0 else []
    
    # Find small components
    small_components = np.where(component_sizes < min_size)[0] + 1  # +1 because background is 0
    
    # Count total voxels in small components
    small_voxels = sum(component_sizes[i-1] for i in small_components) if len(small_components) > 0 else 0
    
    if len(small_components) > 0:
        print(f"Removing {len(small_components)} small components ({small_voxels} voxels)")
        
        # Create mask of components to remove
        remove_mask = np.isin(labeled_volume, small_components)
        
        # Remove small components
        cleaned_volume = binary_volume.copy()
        cleaned_volume[remove_mask] = False
        
        return cleaned_volume
    else:
        print("No small components to remove")
        return binary_volume

def process_ct_scan(points, labels, volume_shape=None, spacing=None, 
                    influence_radius=3.0, closing_kernel_size=3,
                    threshold=0.5, smooth_sigma=1.0,
                    dbscan_eps=0.15, dbscan_min_samples=20,
                    min_component_size=100):
    """
    Process labeled CT scan points to SDF using extrapolation.
    
    Args:
        points: Nx3 array of scan coordinates
        labels: N-length array with 1 for foreground, 0 for background
        volume_shape: Optional. If None, will be determined from points.
        spacing: Optional tuple for CT scan spacing (z, y, x)
        influence_radius: Radius of influence for each point
        closing_kernel_size: Size of kernel for morphological closing
        threshold: Threshold for binarizing the probability volume
        smooth_sigma: Sigma for Gaussian smoothing (0 for no smoothing)
    
    Returns:
        SDF as numpy array, binary volume, and probability volume
    """
    # Apply DBSCAN denoising to remove isolated points
    print("Applying DBSCAN denoising...")
    points, labels = denoise_pc.denoise_point_cloud(
        points, labels, eps=dbscan_eps, min_samples=dbscan_min_samples
    )
    
    # Determine volume shape if not provided
    if volume_shape is None:
        if spacing is not None:
            # Scale by spacing if provided
            scaled_points = points.copy()
            for i in range(3):
                scaled_points[:, i] = scaled_points[:, i] / spacing[i]
            max_coords = np.max(scaled_points, axis=0)
        else:
            max_coords = np.max(points, axis=0)
        
        volume_shape = tuple(np.ceil(max_coords).astype(int) + 1)
        print(f"Automatically determined volume shape: {volume_shape}")
    
    # Create probability volume by extrapolation
    print("Creating extrapolated probability volume...")
    prob_volume = create_extrapolated_volume(
        points, labels, volume_shape, spacing, radius=influence_radius
    )
    
    # Apply morphological closing to get final binary volume
    print("Applying morphological closing...")
    binary_volume = prob_volume > threshold
    if closing_kernel_size > 0:
        kernel = np.ones((closing_kernel_size, closing_kernel_size, closing_kernel_size))
        binary_volume = binary_closing(binary_volume, structure=kernel)
    
    # Remove small disconnected components
    print("Removing small disconnected components...")
    binary_volume = remove_small_components(binary_volume, min_size=min_component_size)
    
    # Compute SDF
    print("Computing SDF...")
    if np.sum(binary_volume) == 0:
        print("Warning: Empty binary volume - SDF calculation may be meaningless")
        return np.ones(binary_volume.shape), binary_volume, prob_volume
    
    if np.all(binary_volume):
        print("Warning: Full binary volume - SDF calculation may be meaningless")
        return -np.ones(binary_volume.shape), binary_volume, prob_volume
    
    print("Ensuring hollow structure...")
    # Erode by 2 voxels to remove interior
    core = binary_erosion(binary_volume, iterations=2)
    hollow_volume = binary_volume & ~core
    
    # Use hollow volume for SDF calculation
    print("Computing SDF from hollow structure...")
    outside_distance = distance_transform_edt(~hollow_volume)
    inside_distance = distance_transform_edt(hollow_volume)
    sdf = outside_distance - inside_distance
    
    # Apply Gaussian smoothing if needed
    if smooth_sigma > 0:
        print(f"Applying Gaussian smoothing with sigma={smooth_sigma}")
        sdf = gaussian_filter(sdf, sigma=smooth_sigma)
    
    return sdf, binary_volume, prob_volume

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
            level = max(sdf_min + 0.1, min(sdf_max - 0.1, 0.0))
            print(f"Using adjusted level: {level}")
        else:
            return None, None, None
    
    try:
        # Extract the isosurface using marching cubes
        kwargs = {'level': level, 'allow_degenerate': False}
        if spacing is not None:
            kwargs['spacing'] = spacing
        
        verts, faces, normals, _ = measure.marching_cubes(sdf, **kwargs)
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

def save_mesh(verts, faces, normals, output_path):
    """
    Save mesh with validation and cleaning.
    
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
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        
        # Clean mesh
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
        
        # Save as PLY
        mesh.export(output_path)
        print(f"Mesh saved successfully to {output_path}")
        return True
    
    except Exception as e:
        print(f"Error saving mesh: {str(e)}")
        return False

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert point cloud to SDF using extrapolation')
    parser.add_argument('points_file', type=str, help='Path to points numpy file (.npy)')
    parser.add_argument('labels_file', type=str, help='Path to labels numpy file (.npy)')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save output files')
    parser.add_argument('--output_prefix', type=str, default='output', help='Prefix for output files')
    parser.add_argument('--volume_shape', type=int, nargs=3, help='Custom volume shape (depth height width)')
    parser.add_argument('--spacing', type=float, nargs=3, help='CT scan spacing (z y x)')
    parser.add_argument('--influence_radius', type=float, default=3.5, help='Radius of influence for each point')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for probability volume')
    parser.add_argument('--closing_size', type=int, default=3, help='Kernel size for morphological closing')
    parser.add_argument('--smooth_sigma', type=float, default=0.3, help='Gaussian smoothing sigma (0 for none)')
    parser.add_argument('--dbscan_eps', type=float, default=10.0, help='DBSCAN epsilon for denoising')
    parser.add_argument('--dbscan_min_samples', type=int, default=5, help='DBSCAN min samples')
    parser.add_argument('--min_component_size', type=int, default=100, help='Minimum size for connected components')
    parser.add_argument('--save_isosurface', action='store_true', help='Save isosurface as PLY mesh')
    
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
    
    # Set parameters
    volume_shape = tuple(args.volume_shape) if args.volume_shape else None
    spacing = tuple(args.spacing) if args.spacing else None
    
    # Process the scan with new method
    print("Processing point cloud...")
    try:
        sdf, binary_volume, prob_volume = process_ct_scan(
            points, labels, 
            volume_shape=volume_shape, 
            spacing=spacing,
            influence_radius=args.influence_radius,
            closing_kernel_size=args.closing_size,
            threshold=args.threshold,
            smooth_sigma=args.smooth_sigma,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            min_component_size=args.min_component_size
        )
    except Exception as e:
        print(f"Error processing scan: {str(e)}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    sdf_path = os.path.join(args.output_dir, f"{args.output_prefix}_sdf.npy")
    volume_path = os.path.join(args.output_dir, f"{args.output_prefix}_binary_volume.npy")
    prob_path = os.path.join(args.output_dir, f"{args.output_prefix}_prob_volume.npy")
    
    print(f"Saving SDF to {sdf_path}")
    np.save(sdf_path, sdf)
    
    print(f"Saving binary volume to {volume_path}")
    np.save(volume_path, binary_volume)
    
    print(f"Saving probability volume to {prob_path}")
    np.save(prob_path, prob_volume)
    
    # Optionally extract and save isosurface
    if args.save_isosurface:
        isosurface_path = os.path.join(args.output_dir, f"{args.output_prefix}_isosurface.ply")
        print(f"Extracting isosurface...")
        
        verts, faces, normals = extract_isosurface(sdf, spacing=spacing)
        if verts is not None:
            save_mesh(verts, faces, normals, isosurface_path)
    
    print("Processing complete!")
    print(f"SDF shape: {sdf.shape}")
    print(f"Min SDF value: {np.min(sdf)}, Max SDF value: {np.max(sdf)}")
    print(f"Binary volume: {np.sum(binary_volume)} foreground voxels out of {binary_volume.size}")

if __name__ == "__main__":
    main()