import numpy as np
import os
from pathlib import Path

def convert_to_ply(points, output_file, color):
    """Helper function to write a PLY file with specified color"""
    header = [
        'ply',
        'format ascii 1.0',
        f'element vertex {len(points)}',
        'property float x',
        'property float y',
        'property float z',
        'property uchar red',
        'property uchar green',
        'property uchar blue',
        'end_header'
    ]
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(header) + '\n')
        for point in points:
            f.write(f'{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n')

def process_directories(point_dir, label_dir, output_dir):
    """
    Process all matching files in point_dir and label_dir to create PLY visualizations
    
    Args:
        point_dir: Directory containing point cloud .npy files
        label_dir: Directory containing label .npy files
        output_dir: Directory where PLY files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of point files
    point_files = sorted(Path(point_dir).glob('*.npy'))
    
    for point_file in point_files:
        # Get corresponding label file (with .npy extension)
        base_name = point_file.stem  # gets name without .npy
        label_file = Path(label_dir) / f"{base_name}.npy"
        
        # Check if label file exists
        if not label_file.exists():
            print(f"Warning: No matching label file for {point_file}")
            continue
            
        print(f"Processing {base_name}...")
        
        try:
            # Load data
            points = np.load(str(point_file))
            labels = np.load(str(label_file))
            
            # Separate points based on labels
            foreground_points = points[labels == 1]
            background_points = points[labels == 0]
            
            # Create output filenames
            fore_output = Path(output_dir) / f"{base_name}_foreground.ply"
            back_output = Path(output_dir) / f"{base_name}_background.ply"
            
            # Save files
            convert_to_ply(foreground_points, str(fore_output), [255, 0, 0])  # Red
            convert_to_ply(background_points, str(back_output), [255, 255, 255])  # White
            
            print(f"Created: {fore_output.name} and {back_output.name}")
            
        except Exception as e:
            print(f"Error processing {base_name}: {str(e)}")

if __name__ == "__main__":
    # Example usage with your actual paths
    point_dir = "../test/inference_res/point"
    label_dir = "../test/inference_res/label"
    output_dir = "../test/visualization_ply"
    
    process_directories(point_dir, label_dir, output_dir)
    print("Processing complete!")