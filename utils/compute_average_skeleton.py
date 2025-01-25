import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
import os
import argparse
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

def load_obj(file_path):
    """Load vertices from an OBJ file"""
    vertices = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):  # vertex data
                coords = line.strip().split()[1:]
                vertices.append([float(x) for x in coords])
    return np.array(vertices)

def uniform_sampling(points, n_samples):
    """
    Perform uniform sampling on point cloud using FPS-like approach
    
    Parameters:
    points: nx3 numpy array of points
    n_samples: number of points to sample
    
    Returns:
    sampled_points: n_samples x 3 numpy array
    """
    if len(points) <= n_samples:
        return points
    
    # Initialize array for sampled points
    sampled_points = np.zeros((n_samples, 3))
    
    # Pick first point randomly
    first_idx = np.random.randint(len(points))
    sampled_points[0] = points[first_idx]
    
    # Use nearest neighbors to implement FPS-like sampling
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(points)
    
    # Iteratively sample points
    for i in range(1, n_samples):
        # Find distances to existing sampled points
        distances, _ = nbrs.kneighbors(sampled_points[:i])
        
        # Find the point that has the maximum distance to its nearest sampled point
        max_idx = np.argmax(np.min(distances, axis=0))
        sampled_points[i] = points[max_idx]
    
    return sampled_points

def normalize_point_cloud(points):
    """
    Center and scale point cloud
    """
    # Center
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # Scale to unit sphere
    scale = np.max(np.linalg.norm(centered, axis=1))
    normalized = centered / scale
    
    return normalized

def align_and_average_point_clouds(point_clouds, n_points=1000):
    """Align and average multiple point clouds using Procrustes analysis"""
    if not point_clouds:
        raise ValueError("No point clouds provided")
    
    # First sample all point clouds to the same number of points
    print(f"\nSampling all point clouds to {n_points} points...")
    sampled_clouds = []
    for i, cloud in enumerate(point_clouds, 1):
        sampled = uniform_sampling(cloud, n_points)
        normalized = normalize_point_cloud(sampled)
        sampled_clouds.append(normalized)
        print(f"Processed cloud {i}/{len(point_clouds)}")
    
    # Use the first point cloud as reference
    reference = sampled_clouds[0]
    
    # Align all point clouds to the reference
    print("\nAligning point clouds...")
    aligned_clouds = []
    for i, cloud in enumerate(sampled_clouds, 1):
        _, transformed, _ = procrustes(reference, cloud)
        aligned_clouds.append(transformed)
        print(f"Aligned cloud {i}/{len(point_clouds)}")
    
    # Convert to matrix format for PCA
    stacked_clouds = np.stack(aligned_clouds)
    
    # Compute average
    average_cloud = np.mean(stacked_clouds, axis=0)
    
    # Compute principal components of variation
    print("\nComputing principal components...")
    pca = PCA()
    reshaped_clouds = stacked_clouds.reshape(len(point_clouds), -1)
    pca.fit(reshaped_clouds)
    
    return (average_cloud, 
            pca.components_.reshape(-1, n_points, 3),
            pca.explained_variance_ratio_)

def save_obj(vertices, file_path):
    """Save vertices to an OBJ file"""
    with open(file_path, 'w') as f:
        for vertex in vertices:
            f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')

def main():
    parser = argparse.ArgumentParser(description='Process OBJ files to create average point cloud')
    parser.add_argument('directory', type=str, help='Directory containing OBJ files')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Directory to save results (default: output)')
    parser.add_argument('--n_points', type=int, default=10000,
                       help='Number of points to sample from each cloud (default: 10000)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all OBJ files
    point_clouds = []
    obj_files = []
    
    print("\nLoading OBJ files...")
    for file in sorted(Path(args.directory).glob('*.obj')):
        try:
            vertices = load_obj(file)
            point_clouds.append(vertices)
            obj_files.append(file.name)
            print(f"Loaded {file.name}: {vertices.shape[0]} vertices")
        except Exception as e:
            print(f"Error loading {file.name}: {e}")
    
    if not point_clouds:
        print("No OBJ files found in directory")
        return
    
    # Process point clouds
    try:
        average_cloud, components, explained_variance = align_and_average_point_clouds(
            point_clouds, n_points=args.n_points)
        
        # Save results
        print("\nSaving results...")
        save_obj(average_cloud, output_dir / 'average_healthy_ribcage.obj')
        
        for i, component in enumerate(components[:3]):
            save_obj(average_cloud + component, 
                    output_dir / f'component_{i+1}_positive.obj')
            save_obj(average_cloud - component, 
                    output_dir / f'component_{i+1}_negative.obj')
        
        with open(output_dir / 'analysis_results.txt', 'w') as f:
            f.write("Processed files:\n")
            for obj_file in obj_files:
                f.write(f"- {obj_file}\n")
            
            f.write("\nExplained variance ratios:\n")
            for i, var in enumerate(explained_variance[:5]):
                f.write(f"Component {i+1}: {var:.4f}\n")
        
        print(f"\nResults saved to {output_dir}")
        print("Files generated:")
        print("- average.obj (average point cloud)")
        print("- component_[1-3]_[positive/negative].obj (principal components)")
        print("- analysis_results.txt (variance explained and file list)")
        
    except Exception as e:
        print(f"Error processing point clouds: {e}")
        raise  # Re-raise the exception to see the full traceback

if __name__ == "__main__":
    main()