import nibabel as nib
import numpy as np
from scipy.spatial import KDTree
import pandas as pd
import argparse
from plyfile import PlyData, PlyElement
import os

def evaluate_fracture_points_from_csv(nifti_path, csv_path, voxel_distance_threshold=0, output_path=None):
    """
    Evaluate if fractures in a CSV file are true positives by checking if their voxel coordinates
    are within a specified threshold distance of labeled fracture voxels in a NIfTI file.
    
    This function is designed to work with CSV files produced by the fracture detection script
    that contain columns like 'point_index', 'x', 'y', 'z', 'component_id', 'rib_designation',
    and 'detection_method'.
    
    Parameters:
    -----------
    nifti_path : str
        Path to the NIfTI label file
    csv_path : str
        Path to the CSV file containing fracture coordinates
    voxel_distance_threshold : int, optional
        Maximum Euclidean distance in voxel space for a point to be considered a true positive.
        If 0, exact voxel coordinate matching is used (default: 0)
    output_path : str, optional
        Path to save the evaluation results
        
    Returns:
    --------
    DataFrame with evaluation results
    """
    print(f"Loading NIfTI file: {nifti_path}")
    nifti = nib.load(nifti_path)
    nifti_data = nifti.get_fdata()
    
    # Find all non-zero voxels (fractures)
    fracture_voxel_indices = np.where(nifti_data > 0)
    
    # Convert to array of [i, j, k] voxel coordinates
    fracture_voxels = np.array(fracture_voxel_indices).T
    
    if len(fracture_voxels) == 0:
        print("No fracture voxels found in the NIfTI file!")
        return None
        
    print(f"Found {len(fracture_voxels)} fracture voxels in the NIfTI file")
    
    print(f"Loading CSV file: {csv_path}")
    try:
        # Load CSV file with fracture coordinates
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
        
    print(f"Evaluating {len(fracture_points)} fracture points from the CSV file")
    
    # Extract coordinates from the points (already in voxel space)
    point_voxel_coords = fracture_points[['x', 'y', 'z']].values
    
    # Round to nearest integer voxel coordinates if needed
    point_voxel_coords_rounded = np.round(point_voxel_coords).astype(int)
    
    # Initialize results arrays
    is_true_positive = np.zeros(len(point_voxel_coords), dtype=bool)
    voxel_distances = np.zeros(len(point_voxel_coords))
    nearest_indices = np.zeros(len(point_voxel_coords), dtype=int)
    
    # Create KDTree for voxel space distance calculation
    voxel_tree = KDTree(fracture_voxels)
    
    # For each point, find nearest fracture voxel and compute distance in voxel space
    for i, voxel_coord in enumerate(point_voxel_coords_rounded):
        # Get distance to nearest fracture voxel (in voxel space)
        distance, nearest_idx = voxel_tree.query([voxel_coord])
        voxel_distances[i] = distance[0]
        nearest_indices[i] = nearest_idx[0]
        
        # Check if distance is within threshold
        is_true_positive[i] = distance[0] <= voxel_distance_threshold
    
    print(f"Evaluation using voxel distance threshold: {voxel_distance_threshold}")
    
    # Create a results dataframe with the original points and evaluation results
    results_df = fracture_points.copy()
    results_df['voxel_x'] = point_voxel_coords_rounded[:, 0]
    results_df['voxel_y'] = point_voxel_coords_rounded[:, 1]
    results_df['voxel_z'] = point_voxel_coords_rounded[:, 2]
    results_df['voxel_distance_to_nearest_fracture'] = voxel_distances
    results_df['nearest_fracture_voxel_index'] = nearest_indices.astype(int)
    results_df['is_true_positive'] = is_true_positive
    
    # Add coordinates of nearest fracture voxel
    nearest_fracture_voxels = fracture_voxels[nearest_indices.astype(int)]
    results_df['nearest_fracture_voxel_x'] = nearest_fracture_voxels[:, 0]
    results_df['nearest_fracture_voxel_y'] = nearest_fracture_voxels[:, 1]
    results_df['nearest_fracture_voxel_z'] = nearest_fracture_voxels[:, 2]
    
    # Compute overall statistics
    tp_count = np.sum(is_true_positive)
    fp_count = len(is_true_positive) - tp_count
    
    print("\n--- Evaluation Results ---")
    print(f"Total fracture points: {len(fracture_points)}")
    print(f"True positives: {tp_count} ({tp_count/len(fracture_points)*100:.2f}%)")
    print(f"False positives: {fp_count} ({fp_count/len(fracture_points)*100:.2f}%)")
    
    # If we have distance data, provide distance statistics
    if 'voxel_distance_to_nearest_fracture' in results_df.columns:
        print(f"Min voxel distance to fracture: {np.min(voxel_distances):.2f}")
        print(f"Max voxel distance to fracture: {np.max(voxel_distances):.2f}")
        print(f"Average voxel distance to fracture: {np.mean(voxel_distances):.2f}")
        
        # Calculate how many fractures fall within each distance range
        distance_bins = [0, 1, 2, 3, 5, 10, np.inf]
        bin_labels = ['0 (exact)', '1', '2', '3-4', '5-9', '10+']
        
        hist, _ = np.histogram(voxel_distances, bins=distance_bins)
        percentages = hist / len(voxel_distances) * 100
        
        print("\nVoxel distance distribution:")
        for i, (count, percentage) in enumerate(zip(hist, percentages)):
            print(f"  Distance {bin_labels[i]}: {count} points ({percentage:.1f}%)")
            
        # Count points with exact match (distance = 0)
        exact_match_count = np.sum(voxel_distances == 0)
        if exact_match_count > 0:
            print(f"  Exact voxel matches: {exact_match_count} ({exact_match_count/len(voxel_distances)*100:.1f}%)")
    
    # If we have rib information, provide statistics per rib
    if 'rib_designation' in points_df.columns:
        print("\n--- Statistics by Rib ---")
        rib_stats = results_df.groupby('rib_designation').agg(
            total_points=('is_true_positive', 'count'),
            true_positives=('is_true_positive', 'sum')
        )
        rib_stats['true_positive_rate'] = rib_stats['true_positives'] / rib_stats['total_points'] * 100
        print(rib_stats)
    
    # If we have detection method information, provide statistics per method
    if 'detection_method' in points_df.columns:
        print("\n--- Statistics by Detection Method ---")
        method_stats = results_df.groupby('detection_method').agg(
            total_points=('is_true_positive', 'count'),
            true_positives=('is_true_positive', 'sum')
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
        is_tp = csv_results['is_true_positive'].values
        
        # Create a structured array for the vertices
        dtype_list = [
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4'),
            ('is_true_positive', 'u1')  # Using uint8 instead of bool for better compatibility
        ]
        
        # Add voxel distance information if available
        if 'voxel_distance_to_nearest_fracture' in csv_results.columns:
            dtype_list.append(('voxel_distance', 'f4'))
        
        vertex_data = np.zeros(len(x), dtype=dtype_list)
        
        # Fill the structured array
        vertex_data['x'] = x
        vertex_data['y'] = y
        vertex_data['z'] = z
        vertex_data['is_true_positive'] = is_tp
        
        # Add distance if available
        if 'voxel_distance_to_nearest_fracture' in csv_results.columns:
            vertex_data['voxel_distance'] = csv_results['voxel_distance_to_nearest_fracture'].values
        
        # Create the vertex element
        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        
        # Create and save the PLY file
        ply_data = PlyData([vertex_element])
        ply_data.write(output_ply_path)
        
        print(f"Evaluation results saved as PLY to {output_ply_path}")
        
    except Exception as e:
        print(f"Error saving evaluation to PLY: {e}")

def process_directories(csv_directory, nifti_directory, voxel_distance_threshold=0, output_directory=None):
    """
    Process matching pairs of CSV and NIfTI files from two directories.
    
    This function matches files by their base name (before the first hyphen).
    For example, "RibFrac1-label.nii" will match with "RibFrac1-rib-cl_pointcloud_endpoints_coordinates.csv"
    
    Parameters:
    -----------
    csv_directory : str
        Path to the directory containing CSV files with fracture coordinates
    nifti_directory : str
        Path to the directory containing NIfTI label files
    voxel_distance_threshold : int, optional
        Maximum Euclidean distance in voxel space for a point to be considered a true positive.
        If 0, exact voxel coordinate matching is used (default: 0)
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
    
    # Dictionary to store rib-specific results for master analysis
    rib_specific_results = {}
    
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
            voxel_distance_threshold=voxel_distance_threshold,
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
                'true_positive_rate': (tp_count / total_count * 100) if total_count > 0 else 0
            })
            
            # Print summary for this ribcage
            print(f"  Total fractures: {total_count}")
            print(f"  True positives: {tp_count} ({tp_count/total_count*100:.2f}%)")
            print(f"  False positives: {total_count - tp_count} ({(total_count-tp_count)/total_count*100:.2f}%)")
            
            # If voxel distance data is available, print distance statistics
            if 'voxel_distance_to_nearest_fracture' in results.columns:
                print(f"  Min distance (voxels): {results['voxel_distance_to_nearest_fracture'].min():.2f}")
                print(f"  Max distance (voxels): {results['voxel_distance_to_nearest_fracture'].max():.2f}")
                print(f"  Average distance (voxels): {results['voxel_distance_to_nearest_fracture'].mean():.2f}")
            
            # Process rib-specific stats if available
            if 'rib_designation' in results.columns:
                rib_stats = results.groupby('rib_designation').agg(
                    total=('is_true_positive', 'count'),
                    tp=('is_true_positive', 'sum')
                )
                rib_stats['tp_rate'] = (rib_stats['tp'] / rib_stats['total'] * 100).round(1)
                
                # Store rib-specific results for the master analysis
                for rib, row in rib_stats.iterrows():
                    if rib not in rib_specific_results:
                        rib_specific_results[rib] = {'total': 0, 'tp': 0}
                    rib_specific_results[rib]['total'] += row['total']
                    rib_specific_results[rib]['tp'] += row['tp']
                
                print("  Rib-specific true positive rates:")
                for rib, row in rib_stats.iterrows():
                    print(f"    {rib}: {row['tp_rate']}% ({row['tp']}/{row['total']})")
    
    # Save overall summary if we have results and an output directory
    if summary_results and output_directory:
        # Basic summary with one row per ribcage
        summary_df = pd.DataFrame(summary_results)
        summary_path = os.path.join(output_directory, "evaluation_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nEvaluation summary saved to {summary_path}")
        
        # Create the master analysis CSV with detailed statistics
        master_analysis_rows = []
        
        # First add the per-ribcage statistics
        for result in summary_results:
            master_analysis_rows.append({
                'ribcage_id': result['ribcage_id'],
                'rib_designation': 'ALL RIBS',
                'total_detections': result['total_points'],
                'true_positives': result['true_positives'],
                'false_positives': result['false_positives'],
                'true_positive_rate': result['true_positive_rate']
            })
        
        # If we have rib-specific data, add per-rib data for each ribcage
        if rib_specific_results:
            # Add rib-specific rows for each ribcage
            for result in summary_results:
                ribcage_id = result['ribcage_id']
                # Load the evaluation CSV to get rib-specific data
                eval_csv_path = os.path.join(output_directory, f"{ribcage_id}_evaluation.csv")
                if os.path.exists(eval_csv_path):
                    eval_df = pd.read_csv(eval_csv_path)
                    if 'rib_designation' in eval_df.columns:
                        rib_stats = eval_df.groupby('rib_designation').agg(
                            total=('is_true_positive', 'count'),
                            tp=('is_true_positive', 'sum')
                        )
                        for rib, row in rib_stats.iterrows():
                            tp_rate = (row['tp'] / row['total'] * 100) if row['total'] > 0 else 0
                            master_analysis_rows.append({
                                'ribcage_id': ribcage_id,
                                'rib_designation': rib,
                                'total_detections': row['total'],
                                'true_positives': row['tp'],
                                'false_positives': row['total'] - row['tp'],
                                'true_positive_rate': tp_rate
                            })
        
        # Add summary statistics across all ribcages
        overall_total = sum(result['total_points'] for result in summary_results)
        overall_tp = sum(result['true_positives'] for result in summary_results)
        overall_fp = sum(result['false_positives'] for result in summary_results)
        overall_tp_rate = (overall_tp / overall_total * 100) if overall_total > 0 else 0
        
        master_analysis_rows.append({
            'ribcage_id': 'ALL RIBCAGES',
            'rib_designation': 'ALL RIBS',
            'total_detections': overall_total,
            'true_positives': overall_tp,
            'false_positives': overall_fp,
            'true_positive_rate': overall_tp_rate
        })
        
        # Add rib-specific summaries across all ribcages
        for rib, stats in rib_specific_results.items():
            tp_rate = (stats['tp'] / stats['total'] * 100) if stats['total'] > 0 else 0
            master_analysis_rows.append({
                'ribcage_id': 'ALL RIBCAGES',
                'rib_designation': rib,
                'total_detections': stats['total'],
                'true_positives': stats['tp'],
                'false_positives': stats['total'] - stats['tp'],
                'true_positive_rate': tp_rate
            })
        
        # Save the master analysis CSV
        master_analysis_df = pd.DataFrame(master_analysis_rows)
        master_analysis_path = os.path.join(output_directory, "master_analysis.csv")
        master_analysis_df.to_csv(master_analysis_path, index=False)
        print(f"Master analysis saved to {master_analysis_path}")
        
        # Print overall statistics
        print("\nOverall Evaluation Results:")
        print(f"Total ribcages processed: {len(summary_df)}")
        print(f"Total fractures evaluated: {overall_total}")
        print(f"Overall true positive rate: {overall_tp_rate:.2f}% ({overall_tp}/{overall_total})")
        
        return summary_df, master_analysis_df
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate fracture locations against NIfTI ground truth')
    parser.add_argument('--csv-dir', help='Directory containing CSV files with fracture coordinates', required=True)
    parser.add_argument('--nifti-dir', help='Directory containing NIfTI label files', required=True)
    parser.add_argument('--threshold', type=int, default=10, 
                    help='Maximum Euclidean distance in voxel space for a true positive. If 0, exact voxel matching is used (default: 0)')
    parser.add_argument('--output-dir', help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Process directories with matching CSV and NIfTI files
    process_directories(
        args.csv_dir,
        args.nifti_dir,
        voxel_distance_threshold=args.threshold,
        output_directory=args.output_dir
    )