import sys
import numpy as np
from pathlib import Path
import open3d as o3d
import argparse
import glob
import os

def normalize_vertices(vertices):
    """Normalize vertex coordinates to range [-1, 1] while preserving aspect ratio"""
    if not vertices:
        return vertices
    
    # Find min and max values for each dimension
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    z_coords = [v[2] for v in vertices]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    z_min, z_max = min(z_coords), max(z_coords)
    
    # Find the largest range to maintain aspect ratio
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = max(x_range, y_range, z_range)
    
    if max_range == 0:
        return vertices
    
    # Normalize vertices
    normalized = []
    for vertex in vertices:
        norm_x = 2 * (vertex[0] - x_min) / max_range - 1
        norm_y = 2 * (vertex[1] - y_min) / max_range - 1
        norm_z = 2 * (vertex[2] - z_min) / max_range - 1
        normalized.append([norm_x, norm_y, norm_z])
    
    return normalized

def compute_normals(vertices, num_neighbors=10, smooth_iterations=0, viewpoint=None):
    """Compute normals for a point cloud using Open3D"""
    # Convert vertices to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(vertices))
    
    # Compute normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=num_neighbors)
    )
    
    # Orient normals towards viewpoint if specified
    if viewpoint is not None:
        pcd.orient_normals_towards_camera_location(viewpoint)
    
    # Apply smoothing iterations if specified
    if smooth_iterations > 0:
        for _ in range(smooth_iterations):
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            smooth_normals = np.asarray(pcd.normals).copy()
            
            for i, point in enumerate(pcd.points):
                [_, idx, _] = pcd_tree.search_knn_vector_3d(point, num_neighbors)
                neighbor_normals = np.asarray(pcd.normals)[idx]
                smooth_normals[i] = np.mean(neighbor_normals, axis=0)
                smooth_normals[i] /= np.linalg.norm(smooth_normals[i])
            
            pcd.normals = o3d.utility.Vector3dVector(smooth_normals)
    
    return np.asarray(pcd.normals).tolist()

def convert_3d_file(input_path, output_path, normalize=True, compute_new_normals=False, num_neighbors=10, smooth_iterations=0, viewpoint=None):
    """Process a single 3D file"""
    vertices = []
    normals = []
    has_normals = False
    
    # Detect file type from extension
    file_type = input_path.split('.')[-1].lower()
    
    # Read file according to its type
    try:
        with open(input_path, 'r') as f:
            if file_type == 'obj':
                for line in f:
                    if line.startswith('v '):
                        coords = line.strip().split()[1:]
                        if len(coords) == 6:
                            vertices.append([float(x) for x in coords[0:3]])
                            normals.append([float(x) for x in coords[3:6]])
                            has_normals = True
                        else:
                            vertices.append([float(x) for x in coords[0:3]])
            elif file_type == 'ply':
                header = True
                vertex_count = 0
                current_vertex = 0
                while header:
                    line = f.readline().strip()
                    if 'element vertex' in line:
                        vertex_count = int(line.split()[-1])
                    elif 'property float nx' in line:
                        has_normals = True
                    elif 'end_header' in line:
                        header = False
                while current_vertex < vertex_count:
                    line = f.readline().strip().split()
                    vertices.append([float(x) for x in line[0:3]])
                    if has_normals:
                        normals.append([float(x) for x in line[3:6]])
                    current_vertex += 1
            elif file_type == 'xyz':
                for line in f:
                    values = line.strip().split()
                    if len(values) >= 6:
                        vertices.append([float(x) for x in values[0:3]])
                        normals.append([float(x) for x in values[3:6]])
                        has_normals = True
                    elif len(values) >= 3:
                        vertices.append([float(x) for x in values[0:3]])
        
        # Normalize vertices if requested
        if normalize:
            vertices = normalize_vertices(vertices)
            print(f"Vertices normalized for {input_path}")
        
        # Compute new normals if requested or if no normals exist
        if compute_new_normals or not has_normals:
            print(f"Computing normals for {input_path}...")
            normals = compute_normals(vertices, num_neighbors, smooth_iterations, viewpoint)
            has_normals = True
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write output file
        with open(output_path, 'w') as f:
            if has_normals:
                for v, n in zip(vertices, normals):
                    f.write(f"{v[0]} {v[1]} {v[2]} {n[0]} {n[1]} {n[2]}\n")
            else:
                for v in vertices:
                    f.write(f"{v[0]} {v[1]} {v[2]}\n")

        print(f"Processed {input_path} → {output_path} ({len(vertices)} vertices)")
        return True
    
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def process_directory(input_dir, output_dir, normalize=True, compute_new_normals=False, 
                     num_neighbors=10, smooth_iterations=0, viewpoint=None):
    """Process all supported files in input directory"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported file extensions
    supported_formats = ['.obj', '.ply', '.xyz']
    
    # Find all supported files in input directory and subdirectories
    all_files = []
    for ext in supported_formats:
        all_files.extend(list(input_dir.glob(f'**/*{ext}')))
    
    if not all_files:
        print(f"No supported files found in {input_dir}")
        return
    
    print(f"Found {len(all_files)} files to process")
    
    # Process each file
    successful = 0
    for input_file in all_files:
        # Create relative path for output file
        rel_path = input_file.relative_to(input_dir)
        output_path = output_dir / rel_path.with_suffix('.xyz')
        
        # Process the file
        if convert_3d_file(
            str(input_file),
            str(output_path),
            normalize=normalize,
            compute_new_normals=compute_new_normals,
            num_neighbors=num_neighbors,
            smooth_iterations=smooth_iterations,
            viewpoint=viewpoint
        ):
            successful += 1
    
    print(f"\nProcessing complete: {successful}/{len(all_files)} files processed successfully")

def main():
    parser = argparse.ArgumentParser(description='Convert and process 3D point cloud files')
    parser.add_argument('input_dir', help='Input directory containing 3D files')
    parser.add_argument('output_dir', help='Output directory for processed files')
    parser.add_argument('--no-normalize', action='store_true', help='Disable vertex normalization')
    parser.add_argument('--compute-normals', action='store_true', help='Compute new normals (ignores existing normals)')
    parser.add_argument('--neighbors', type=int, default=10, help='Number of neighbors for normal estimation')
    parser.add_argument('--smooth', type=int, default=0, help='Number of normal smoothing iterations')
    parser.add_argument('--viewpoint', type=float, nargs=3, help='Viewpoint position (x y z)')
    
    args = parser.parse_args()
    
    # Verify input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)
    
    try:
        process_directory(
            args.input_dir,
            args.output_dir,
            normalize=not args.no_normalize,
            compute_new_normals=args.compute_normals,
            num_neighbors=args.neighbors,
            smooth_iterations=args.smooth,
            viewpoint=args.viewpoint
        )
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()  