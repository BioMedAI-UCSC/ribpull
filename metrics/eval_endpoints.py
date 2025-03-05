import nibabel as nib
import numpy as np
from scipy.spatial import KDTree
import pandas as pd
import argparse
from plyfile import PlyData, PlyElement
import os

def evaluate_fracture_points_from_csv(nifti_path, csv_path, distance_threshold, output_path=None):
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

def process_directories(csv_directory, nifti_directory, distance_threshold=5.0, output_directory=None):
    """
    Process matching pairs of CSV and NIfTI files from two directories.
    
    This function matches files by their base name (before the first hyphen).
    For example, "RibFrac1-label.nii" will match with "RibFrac1-rib-cl_pointcloud_endpoints_coordinates.csv"
    
    Parameters:
    -----------
    csv_directory : str
        Path to the directory containing CSV files with endpoint coordinates
    nifti_directory : str
        Path to the directory containing NIfTI label files
    distance_threshold : float
        Maximum distance (mm) for a point to be considered a true positive
    output_directory : str, optional
        Directory to save output files
    """
    # Create output directory if specified
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Get list of files in each directory
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    nifti_files = [f for f in os.listdir(nifti_directory) if f.endswith(('.nii', '.nii.gz'))]
    
    print(f"Found {len(csv_files)} CSV files and {len(nifti_files)} NIfTI files")
    
    # Create dictionaries mapping base names to files
    csv_dict = {}
    for file in csv_files:
        # Extract base name (before first hyphen)
        base_name = file.split('-')[0]
        csv_dict[base_name] = file
    
    nifti_dict = {}
    for file in nifti_files:
        # Extract base name (before first hyphen)
        base_name = file.split('-')[0]
        nifti_dict[base_name] = file
    
    # Find matching pairs
    matched_pairs = []
    for base_name in csv_dict.keys():
        if base_name in nifti_dict:
            matched_pairs.append((base_name, csv_dict[base_name], nifti_dict[base_name]))
    
    print(f"Found {len(matched_pairs)} matching file pairs")
    
    # Create a summary dataframe
    summary_results = []
    
    # Process each matching pair
    for base_name, csv_file, nifti_file in matched_pairs:
        print(f"\nProcessing: {base_name}")
        print(f"  CSV: {csv_file}")
        print(f"  NIfTI: {nifti_file}")
        
        csv_path = os.path.join(csv_directory, csv_file)
        nifti_path = os.path.join(nifti_directory, nifti_file)
        
        # Set up output paths if needed
        if output_directory:
            output_csv = os.path.join(output_directory, f"{base_name}_evaluation.csv")
            output_ply = os.path.join(output_directory, f"{base_name}_evaluation.ply")
        else:
            output_csv = None
            output_ply = None
        
        # Process the pair
        results = evaluate_fracture_points_from_csv(
            nifti_path,
            csv_path,
            distance_threshold=distance_threshold,
            output_path=output_csv
        )
        
        if results is not None:
            # Save PLY visualization if requested
            if output_directory:
                save_evaluation_to_ply(results, output_ply)
            
            # Calculate summary statistics
            tp_count = results['is_true_positive'].sum()
            total_count = len(results)
            
            # Store summary statistics
            summary_results.append({
                'ribcage_id': base_name,
                'csv_file': csv_file,
                'nifti_file': nifti_file,
                'total_points': total_count,
                'true_positives': tp_count,
                'false_positives': total_count - tp_count,
                'true_positive_rate': (tp_count / total_count * 100) if total_count > 0 else 0,
                'avg_distance': results['distance_to_nearest_fracture'].mean()
            })
            
            # Print summary for this ribcage
            print(f"  Total endpoints: {total_count}")
            print(f"  True positives: {tp_count} ({tp_count/total_count*100:.2f}%)")
            print(f"  False positives: {total_count - tp_count} ({(total_count-tp_count)/total_count*100:.2f}%)")
            print(f"  Average distance: {results['distance_to_nearest_fracture'].mean():.2f} mm")
            
            # Print rib-specific stats if available
            if 'rib_designation' in results.columns:
                rib_stats = results.groupby('rib_designation').agg(
                    total=('is_true_positive', 'count'),
                    tp=('is_true_positive', 'sum')
                )
                rib_stats['tp_rate'] = (rib_stats['tp'] / rib_stats['total'] * 100).round(1)
                print("  Rib-specific true positive rates:")
                for rib, row in rib_stats.iterrows():
                    print(f"    {rib}: {row['tp_rate']}% ({row['tp']}/{row['total']})")
    
    # Save overall summary if we have results and an output directory
    if summary_results and output_directory:
        summary_df = pd.DataFrame(summary_results)
        summary_path = os.path.join(output_directory, "evaluation_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nEvaluation complete. Summary saved to {summary_path}")
        
        # Print overall statistics
        overall_tp = summary_df['true_positives'].sum()
        overall_total = summary_df['total_points'].sum()
        print("\nOverall Evaluation Results:")
        print(f"Total ribcages processed: {len(summary_df)}")
        print(f"Total endpoints evaluated: {overall_total}")
        print(f"Overall true positive rate: {overall_tp/overall_total*100:.2f}% ({overall_tp}/{overall_total})")
        
        return summary_df
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate fracture endpoints against NIfTI ground truth')
    parser.add_argument('--csv-dir', help='Directory containing CSV files with endpoint coordinates', required=True)
    parser.add_argument('--nifti-dir', help='Directory containing NIfTI label files', required=True)
    parser.add_argument('--threshold', type=float, default=5.0, 
                        help='Maximum distance (mm) for a true positive')
    parser.add_argument('--output-dir', help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Process directories with matching CSV and NIfTI files
    process_directories(
        args.csv_dir,
        args.nifti_dir,
        distance_threshold=args.threshold,
        output_directory=args.output_dir
    )