import nibabel as nib
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add backwards compatibility for older nibabel versions
if not hasattr(np, 'float'):
    np.float = float

def nifti_to_pointcloud(nifti_path, threshold=0):
    """
    Convert a NIfTI file to a point cloud.
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()
    spacing = img.header.get_zooms()[:3]
    
    binary_data = data > threshold
    points = np.array(np.where(binary_data)).T
    points = points * spacing
    
    return points

def save_pointcloud_obj(points, output_path):
    """
    Save point cloud to OBJ file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for point in points:
            f.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        
        for i in range(1, len(points) + 1):
            f.write(f"p {i}\n")

def process_nifti_files(input_dir, output_dir, threshold=None):
    """
    Process all NIfTI files in the input directory.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .nii.gz files
    nifti_files = list(input_dir.glob("*.nii.gz"))
    
    if not nifti_files:
        print("No .nii.gz files found!")
        return
    
    print(f"Found {len(nifti_files)} NIfTI files to process")
    
    # Prepare summary dictionary
    summary = {
        'processed_on': datetime.now().isoformat(),
        'input_directory': str(input_dir),
        'output_directory': str(output_dir),
        'processed_files': {}
    }
    
    # Process each file
    for nifti_file in nifti_files:
        print(f"\nProcessing {nifti_file.name}")
        
        try:
            # Convert to point cloud
            points = nifti_to_pointcloud(nifti_file, threshold=threshold)
            
            # Create output filename
            output_file = output_dir / f"{nifti_file.stem.replace('.nii', '')}_pointcloud.obj"
            
            # Save point cloud
            save_pointcloud_obj(points, output_file)
            print(f"Point cloud saved with {len(points)} points")
            
            # Collect statistics
            stats = {
                'points': len(points),
                'threshold_used': float(threshold) if threshold is not None else 0
            }
            
            if len(points) > 0:
                stats['bounds'] = {
                    'min': points.min(axis=0).tolist(),
                    'max': points.max(axis=0).tolist()
                }
            
            # Add to summary
            summary['processed_files'][nifti_file.name] = stats
            
        except Exception as e:
            print(f"Error processing {nifti_file.name}: {str(e)}")
            summary['processed_files'][nifti_file.name] = {'error': str(e)}
    
    # Save summary
    with open(output_dir / 'conversion_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nProcessing complete. Summary saved to: {output_dir / 'conversion_summary.json'}")

def main():
    parser = argparse.ArgumentParser(description="Convert NIfTI files to point cloud OBJ files")
    parser.add_argument("input_dir", help="Directory containing .nii.gz files")
    parser.add_argument("output_dir", help="Directory for output files")
    parser.add_argument("--threshold", type=float, help="Optional: threshold value (default: 0)", default=0)
    
    args = parser.parse_args()
    process_nifti_files(args.input_dir, args.output_dir, args.threshold)

if __name__ == "__main__":
    main()
    