import nibabel as nib
import numpy as np
from scipy.spatial import KDTree
import pandas as pd
import argparse
from plyfile import PlyData, PlyElement
import os

def evaluate_fracture_points_from_csv(nifti_path, csv_path, distance_threshold=5.0, output_path=None):
    """
    Evaluate if fracture endpoints in a CSV file are true positives by checking their proximity
    to labeled fracture voxels in a NIfTI file.
    
    This function is designed to work with CSV files produced by the endpoint detection script
    that contain columns like 'point_index', 'x', 'y', 'z', 'component_id', 'rib_designation',
    and 'detection_method'.
    
    Parameters:
    -----------
    nifti_path : str
        Path to the NIfTI label file
    csv_path : str
        Path to the CSV file containing endpoint coordinates
    distance_threshold : float
        Maximum distance (in mm) for a point to be considered a true positive
    output_path : str, optional
        Path to save the evaluation results
        
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
    
    print(f"Loading CSV file: {csv_path}")
    try:
        # Load CSV file with endpoint coordinates
        points_df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        required_columns = ['x', 'y', 'z']
        for col in required_columns:
            if col not in points_df.columns:
                print(f"Error: Required column '{col}' not found in CSV file")
                return None
        
        # Display available columns for user information
        print(f"Available columns in CSV: {list(points_df.columns)}")
        
        # Create a new column for the rib designation if component_id and rib_designation exist
        if 'component_id' in points_df.columns and 'rib_designation' in points_df.columns:
            points_df['rib_component'] = points_df['rib_designation'] + "-" + points_df['component_id'].astype(str)
        
        # Check if potential fracture information exists in the CSV
        fracture_col = None
        for col in ['is_fracture', 'fracture_detected', 'potential_fracture']:
            if col in points_df.columns:
                fracture_col = col
                print(f"Using '{fracture_col}' as fracture indicator column")
                break
                
        # If no fracture column found, add one assuming all points are potential fractures
        if fracture_col is None:
            print("No fracture indicator column found. Assuming all points are fracture candidates.")
            points_df['is_fracture'] = 1
            fracture_col = 'is_fracture'
            
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None
    
    # Filter to only get the fracture points
    if fracture_col == 'is_fracture':
        fracture_points = points_df
    else:
        fracture_points = points_df[points_df[fracture_col] > 0]
    
    if len(fracture_points) == 0:
        print("No fracture points found in the CSV file!")
        return None
        
    print(f"Evaluating {len(fracture_points)} endpoint points from the CSV file")
    
    # Extract coordinates from the points
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
    print(f"Total endpoint points: {len(fracture_points)}")
    print(f"True positives: {tp_count} ({tp_count/len(fracture_points)*100:.2f}%)")
    print(f"False positives: {fp_count} ({fp_count/len(fracture_points)*100:.2f}%)")
    
    # Calculate minimum, maximum, and average distances
    print(f"Min distance to fracture: {np.min(distances):.2f} mm")
    print(f"Max distance to fracture: {np.max(distances):.2f} mm")
    print(f"Average distance to fracture: {np.mean(distances):.2f} mm")
    
    # If we have rib information, provide statistics per rib
    if 'rib_designation' in points_df.columns:
        print("\n--- Statistics by Rib ---")
        rib_stats = results_df.groupby('rib_designation').agg(
            total_points=('is_true_positive', 'count'),
            true_positives=('is_true_positive', 'sum'),
            avg_distance=('distance_to_nearest_fracture', 'mean')
        )
        rib_stats['true_positive_rate'] = rib_stats['true_positives'] / rib_stats['total_points'] * 100
        print(rib_stats)
    
    # If we have detection method information, provide statistics per method
    if 'detection_method' in points_df.columns:
        print("\n--- Statistics by Detection Method ---")
        method_stats = results_df.groupby('detection_method').agg(
            total_points=('is_true_positive', 'count'),
            true_positives=('is_true_positive', 'sum'),
            avg_distance=('distance_to_nearest_fracture', 'mean')
        )
        method_stats['true_positive_rate'] = method_stats['true_positives'] / method_stats['total_points'] * 100
        print(method_stats)
    
    # Save results if output path provided
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    return results_df

def save_evaluation_to_ply(csv_results, output_ply_path):
    """
    Save the evaluation results to a PLY file for visualization.
    
    Parameters:
    -----------
    csv_results : DataFrame
        DataFrame with evaluation results
    output_ply_path : str
        Path to save the PLY file
    """
    try:
        # Extract the coordinates
        x = csv_results['x'].values
        y = csv_results['y'].values
        z = csv_results['z'].values
        
        # Extract evaluation results
        distance = csv_results['distance_to_nearest_fracture'].values
        is_tp = csv_results['is_true_positive'].values
        
        # Create a structured array for the vertices
        vertex_data = np.zeros(len(x), dtype=[
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4'),
            ('distance_to_fracture', 'f4'),
            ('is_true_positive', 'u1')  # Using uint8 instead of bool for better compatibility
        ])
        
        # Fill the structured array
        vertex_data['x'] = x
        vertex_data['y'] = y
        vertex_data['z'] = z
        vertex_data['distance_to_fracture'] = distance
        vertex_data['is_true_positive'] = is_tp
        
        # Create the vertex element
        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        
        # Create and save the PLY file
        ply_data = PlyData([vertex_element])
        ply_data.write(output_ply_path)
        
        print(f"Evaluation results saved as PLY to {output_ply_path}")
        
    except Exception as e:
        print(f"Error saving evaluation to PLY: {e}")

def evaluate_single_ribcage(nifti_path, csv_path, distance_threshold=5.0, output_path=None, output_ply=None):
    """
    Evaluate a single ribcage by processing a CSV file containing endpoints for that ribcage.
    
    Parameters:
    -----------
    nifti_path : str
        Path to the NIfTI label file for this specific ribcage
    csv_path : str
        Path to the CSV file with endpoint coordinates for this ribcage
    distance_threshold : float
        Maximum distance (mm) for a point to be considered a true positive
    output_path : str, optional
        Path to save the evaluation results CSV
    output_ply : str, optional
        Path to save the PLY visualization
    
    Returns:
    --------
    DataFrame with evaluation results
    """
    print(f"\nEvaluating single ribcage")
    print(f"NIfTI file: {nifti_path}")
    print(f"CSV file: {csv_path}")
    
    # Evaluate the endpoints
    results = evaluate_fracture_points_from_csv(
        nifti_path,
        csv_path,
        distance_threshold=distance_threshold,
        output_path=output_path
    )
    
    if results is not None:
        # Calculate and print overall ribcage statistics
        tp_count = results['is_true_positive'].sum()
        total_count = len(results)
        
        print("\n--- Ribcage Summary ---")
        print(f"Total endpoints: {total_count}")
        print(f"True positives: {tp_count} ({tp_count/total_count*100:.2f}%)")
        print(f"False positives: {total_count - tp_count} ({(total_count-tp_count)/total_count*100:.2f}%)")
        print(f"Average distance to fracture: {results['distance_to_nearest_fracture'].mean():.2f} mm")
        
        # Save to PLY if requested
        if output_ply:
            save_evaluation_to_ply(results, output_ply)
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate fracture endpoints from CSV against NIfTI labels')
    parser.add_argument('--nifti', required=True, help='Path to the NIfTI label file')
    parser.add_argument('--csv', required=True, help='Path to the CSV file with endpoint coordinates for a single ribcage')
    parser.add_argument('--threshold', type=float, default=5.0, 
                        help='Maximum distance (mm) for a true positive (default: 5.0)')
    parser.add_argument('--output', help='Path to save the evaluation results CSV')
    parser.add_argument('--output-ply', help='Path to save visualization PLY file')
    
    args = parser.parse_args()
    
    # Process a single ribcage
    evaluate_single_ribcage(
        args.nifti,
        args.csv,
        distance_threshold=args.threshold,
        output_path=args.output,
        output_ply=args.output_ply
    )