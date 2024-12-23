import numpy as np
import os

def save_points_to_obj(points, output_path):
    """
    Save point cloud data to OBJ format
    
    Args:
        points: Numpy array of points
        output_path: Path where to save the .obj file
    """
    with open(output_path, 'w') as f:
        # Write vertices
        for point in points:
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")

def save_labels_to_obj(points, labels, output_path):
    """
    Save labeled point cloud data to OBJ format with colors
    
    Args:
        points: Numpy array of points
        labels: Numpy array of labels
        output_path: Path where to save the labeled .obj file
    """
    with open(output_path, 'w') as f:
        # Write vertices
        for point, label in zip(points, labels):
            # Define colors based on labels (modify as needed)
            color = [1, 0, 0] if label == 0 else [0, 1, 0]  # Red for 0, Green for 1
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")
            f.write(f"#vc {color[0]} {color[1]} {color[2]}\n")

def batch_convert_directory():
    """Convert all point clouds in the inference_res directory"""
    points_dir = "../test/inference_res/point/"
    labels_dir = "../test/inference_res/label/"
    output_dir = "../test/inference_res/obj/"
    
    # Create output directories
    points_output_dir = os.path.join(output_dir, "points")
    labels_output_dir = os.path.join(output_dir, "labels")
    os.makedirs(points_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)
    
    # Process each file
    for filename in os.listdir(points_dir):
        if filename.endswith('.npy'):
            points_path = os.path.join(points_dir, filename)
            labels_path = os.path.join(labels_dir, filename[:-4] + '.npy')
            
            # Output paths for separate files
            points_output_path = os.path.join(points_output_dir, filename[:-4] + '.obj')
            labels_output_path = os.path.join(labels_output_dir, filename[:-4] + '_labeled.obj')
            
            print(f"Converting {filename}...")
            
            # Load data
            points = np.load(points_path)
            labels = np.load(labels_path)
            
            # Save separate OBJ files
            save_points_to_obj(points, points_output_path)
            save_labels_to_obj(points, labels, labels_output_path)
            
            print(f"Saved points to {points_output_path}")
            print(f"Saved labeled points to {labels_output_path}")

if __name__ == "__main__":
    batch_convert_directory()