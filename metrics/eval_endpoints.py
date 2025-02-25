import nibabel as nib
import numpy as np
from scipy.spatial import KDTree
import pandas as pd
import argparse
from plyfile import PlyData, PlyElement

def evaluate_fracture_points(nifti_path, points_path, distance_threshold=5.0, output_path=None, fracture_property=None):
    """
Evaluate if points in a PLY point cloud are true positives by checking their proximity
    to labeled fracture voxels in a NIfTI file.
    
    Parameters:
    -----------
    nifti_path : str
        Path to the NIfTI label file
    points_path : str
        Path to the PLY point cloud file
    distance_threshold : float
        Maximum distance (in mm) for a point to be considered a true positive
    output_path : str, optional
        Path to save the evaluation results
    fracture_property : str, optional
        Name of the property in the PLY file that indicates fracture status
        
    Returns:
    --------
    DataFrame with evaluation results
    """
    print(f"Loading NIfTI file: {nifti_path}")
    nifti = nib.load(nifti_path)
    nifti_data = nifti.get_fdata()
    affine = nifti.affine
    
    voxel_size = nifti.header.get_zooms()
    print(f"Voxel dimensions (mm): {voxel_size}")
    
    # Find all non-zero voxels (fractures)
    fracture_voxel_indices = np.where(nifti_data > 0)
    
    # Convert voxel indices to world coordinates
    fracture_voxels = np.array(fracture_voxel_indices).T
    
    if len(fracture_voxels) == 0:
        print("No fracture voxels found in the NIfTI file!")
        return None
        
    print(f"Found {len(fracture_voxels)} fracture voxels in the NIfTI file")
    
    # Transform voxel indices to world coordinates using the affine matrix
    fracture_coords = nib.affines.apply_affine(affine, fracture_voxels)
    
    # Create a KD-Tree for efficient nearest neighbor search
    fracture_tree = KDTree(fracture_coords)
    
    print(f"Loading PLY point cloud: {points_path}")
    try:
        ply_data = PlyData.read(points_path)
        vertex_data = ply_data['vertex']
        
        x = vertex_data['x']
        y = vertex_data['y']
        z = vertex_data['z']
        
        # Get available properties in the PLY file
        property_names = [p.name for p in vertex_data.properties]
        print(f"Available properties in PLY: {property_names}")
        
        # Create a DataFrame from the PLY data
        points_df = pd.DataFrame({
            'x': x,
            'y': y,
            'z': z
        })
        
        # Add all properties from the PLY file to the DataFrame
        for prop in property_names:
            if prop not in ['x', 'y', 'z']:
                points_df[prop] = vertex_data[prop]
        
        # If fracture property is specified, use it
        if fracture_property is not None:
            if fracture_property in property_names:
                fracture_col = fracture_property
            else:
                print(f"Warning: Specified fracture property '{fracture_property}' not found in PLY file.")
                fracture_col = None
        else:
            # Try to find a property that might indicate fracture status
            fracture_col = None
            for prop in ['fracture', 'is_fracture', 'label', 'class', 'classification']:
                if prop in property_names:
                    fracture_col = prop
                    print(f"Using '{fracture_col}' as fracture indicator property")
                    break
                    
        if fracture_col is None:
            print("Warning: No fracture label property found. Assuming all points are fracture candidates.")
            points_df['fracture'] = 1
            fracture_col = 'fracture'
            
    except Exception as e:
        print(f"Error loading PLY file: {e}")
        return None
    
    # Filter to only get the fracture points from the point cloud
    fracture_points = points_df[points_df[fracture_col] > 0]
    
    if len(fracture_points) == 0:
        print("No fracture points found in the point cloud!")
        return None
        
    print(f"Evaluating {len(fracture_points)} fracture points from the point cloud")
    
    # Extract coordinates from the point cloud
    point_coords = fracture_points[['x', 'y', 'z']].values
    
    # Find the distance to the nearest fracture voxel for each point
    distances, nearest_voxel_indices = fracture_tree.query(point_coords)
    
    # Determine true positives (points close enough to a fracture voxel)
    is_true_positive = distances <= distance_threshold
    
    # Create a results dataframe with the original points and evaluation results
    results_df = fracture_points.copy()
    results_df['distance_to_nearest_fracture'] = distances
    results_df['nearest_fracture_voxel_index'] = nearest_voxel_indices
    results_df['is_true_positive'] = is_true_positive
    
    # Get the coordinates of the nearest fracture voxel for each point
    nearest_fractures = fracture_coords[nearest_voxel_indices]
    results_df['nearest_fracture_x'] = nearest_fractures[:, 0]
    results_df['nearest_fracture_y'] = nearest_fractures[:, 1]
    results_df['nearest_fracture_z'] = nearest_fractures[:, 2]
    
    # Compute overall statistics
    tp_count = np.sum(is_true_positive)
    fp_count = len(is_true_positive) - tp_count
    
    print("\n--- Evaluation Results ---")
    print(f"Total fracture points: {len(fracture_points)}")
    print(f"True positives: {tp_count} ({tp_count/len(fracture_points)*100:.2f}%)")
    print(f"False positives: {fp_count} ({fp_count/len(fracture_points)*100:.2f}%)")
    
    # Calculate minimum, maximum, and average distances
    print(f"Min distance to fracture: {np.min(distances):.2f} mm")
    print(f"Max distance to fracture: {np.max(distances):.2f} mm")
    print(f"Average distance to fracture: {np.mean(distances):.2f} mm")
    
    # Save results if output path provided
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    return results_df

def save_annotated_ply(original_ply_path, results_df, output_ply_path):
    try:
        # Load the original PLY file
        ply_data = PlyData.read(original_ply_path)
        vertex_data = ply_data['vertex']
        
        # Create a dictionary to map from index to the new properties
        # Using a dictionary as the indices of results_df may not match the original PLY indices
        distance_map = {}
        tp_map = {}
        
        for idx, row in results_df.iterrows():
            # Create a key from the x,y,z coordinates for matching
            key = (row['x'], row['y'], row['z'])
            distance_map[key] = row['distance_to_nearest_fracture']
            tp_map[key] = row['is_true_positive']
        
        # Create new property arrays
        n_vertices = len(vertex_data)
        distance_array = np.zeros(n_vertices, dtype='float32')
        tp_array = np.zeros(n_vertices, dtype='bool')
        
        # Fill in the new property arrays
        for i in range(n_vertices):
            key = (vertex_data['x'][i], vertex_data['y'][i], vertex_data['z'][i])
            if key in distance_map:
                distance_array[i] = distance_map[key]
                tp_array[i] = tp_map[key]
        
        # Create a new vertex element with the additional properties
        old_properties = vertex_data.properties
        
        # Create property list with evaluation results
        new_properties = list(old_properties)
        new_properties.append(('distance_to_fracture', 'float32'))
        new_properties.append(('is_true_positive', 'bool'))
        
        # Create a new vertex array with all data
        vertex_arrays = []
        for prop in vertex_data.properties:
            vertex_arrays.append(vertex_data[prop.name])
        
        # Add new arrays
        vertex_arrays.append(distance_array)
        vertex_arrays.append(tp_array)
        
        # Create the new vertex element
        vertex_dtype = [(p.name, p.val_dtype) for p in vertex_data.properties]
        vertex_dtype.append(('distance_to_fracture', 'float32'))
        vertex_dtype.append(('is_true_positive', 'bool'))
        
        # Pack the data into a structured array
        vertex_data_combined = np.zeros(n_vertices, dtype=vertex_dtype)
        for i, name in enumerate([p.name for p in vertex_data.properties]):
            vertex_data_combined[name] = vertex_arrays[i]
        
        vertex_data_combined['distance_to_fracture'] = distance_array
        vertex_data_combined['is_true_positive'] = tp_array
        
        # Create the new PLY element
        vertex_element = PlyElement.describe(vertex_data_combined, 'vertex')
        
        # Create the new PLY data with the same header as the original but with our new vertex element
        other_elements = [e for e in ply_data.elements if e.name != 'vertex']
        new_elements = [vertex_element] + other_elements
        
        # Create and save the new PLY file
        new_ply_data = PlyData(new_elements, ply_data.comments, ply_data.obj_info)
        new_ply_data.write(output_ply_path)
        
        print(f"Annotated PLY saved to {output_ply_path}")
        
    except Exception as e:
        print(f"Error saving annotated PLY: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate fracture points against NIfTI labels')
    parser.add_argument('--nifti', required=True, help='Path to the NIfTI label file')
    parser.add_argument('--points', required=True, help='Path to the PLY point cloud file')
    parser.add_argument('--threshold', type=float, default=5.0, 
                        help='Maximum distance (mm) for a true positive (default: 5.0)')
    parser.add_argument('--output', help='Path to save the evaluation results CSV')
    parser.add_argument('--output-ply', help='Path to save the annotated PLY file')
    parser.add_argument('--fracture-property', help='Name of the property in the PLY file that indicates fracture status')
    
    args = parser.parse_args()
    
    results = evaluate_fracture_points(
        args.nifti, 
        args.points, 
        distance_threshold=args.threshold,
        output_path=args.output,
        fracture_property=args.fracture_property
    )
    
    if results is not None and args.output_ply:
        save_annotated_ply(args.points, results, args.output_ply)