import numpy as np
from scipy.stats import multivariate_normal
import trimesh

def load_obj(file_path):
    """
    Load an OBJ file and return vertices
    """
    vertices = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                coords = line.split()[1:]
                vertices.append([float(x) for x in coords])
    raw_vertices = np.array(vertices)
    mesh = trimesh.load_mesh(file_path)
    return raw_vertices, mesh.faces

def save_obj(file_path, vertices, faces=None):
    """
    Save vertices to an OBJ file as a point cloud.
    
    Args:
        file_path (str): Output path for the OBJ file
        vertices (numpy.ndarray): Array of vertex coordinates with shape (N, 3)
        faces (None): Kept for compatibility but not used
    """
    with open(file_path, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f'v {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n')

def add_gaussian_noise(points, std_dev=0.01):
    """
    Add Gaussian noise to each point independently
    """
    noise = np.random.normal(0, std_dev, points.shape)
    return points + noise

def add_salt_and_pepper_noise(points, noise_ratio=0.05, displacement=0.1):
    """
    Add salt and pepper noise by randomly displacing some points
    """
    noisy_points = points.copy()
    num_points = len(points)
    num_noise_points = int(num_points * noise_ratio)
    
    noise_indices = np.random.choice(num_points, num_noise_points, replace=False)
    
    random_directions = np.random.randn(num_noise_points, 3)
    random_directions /= np.linalg.norm(random_directions, axis=1)[:, np.newaxis]
    random_magnitudes = np.random.uniform(0, displacement, num_noise_points)
    
    noisy_points[noise_indices] += random_directions * random_magnitudes[:, np.newaxis]
    return noisy_points

def add_structured_noise(points, frequency=5, amplitude=0.01):
    """
    Add structured sinusoidal noise
    """
    x_noise = amplitude * np.sin(frequency * points[:, 0])
    y_noise = amplitude * np.cos(frequency * points[:, 1])
    z_noise = amplitude * np.sin(frequency * points[:, 2])
    
    noise = np.stack([x_noise, y_noise, z_noise], axis=1)
    return points + noise

def add_clustered_noise(points, num_clusters=5, cluster_std=0.05, points_per_cluster=20):
    """
    Add clustered noise points around random locations
    """
    cluster_centers = points[np.random.choice(len(points), num_clusters)]
    
    noise_points = []
    for center in cluster_centers:
        cluster_points = multivariate_normal.rvs(
            mean=center, 
            cov=cluster_std**2 * np.eye(3), 
            size=points_per_cluster
        )
        noise_points.append(cluster_points)
    
    return np.vstack([points] + noise_points)

def process_obj_with_noise(input_path, output_path, noise_type='gaussian', **noise_params):
    """
    Load OBJ, add noise, and save the result
    
    Args:
        input_path: path to input OBJ file
        output_path: path to save noisy OBJ file
        noise_type: 'gaussian', 'salt_pepper', 'structured', or 'clustered'
        **noise_params: parameters for the specific noise function
    """
    # Load the OBJ file
    vertices, faces = load_obj(input_path)
    # Using Gaussian for now, but other types of noise can be applied
    if noise_type == 'gaussian':
        noisy_vertices = add_gaussian_noise(vertices, **noise_params)
    elif noise_type == 'salt_pepper':
        noisy_vertices = add_salt_and_pepper_noise(vertices, **noise_params)
    elif noise_type == 'structured':
        noisy_vertices = add_structured_noise(vertices, **noise_params)
    elif noise_type == 'clustered':
        noisy_vertices = add_clustered_noise(vertices, **noise_params)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Save the noisy mesh
    save_obj(output_path, noisy_vertices, faces)

if __name__ == "__main__":
    input_file = "../test/fromSkeletonLearning/bitore_skel.obj"
    output_file = "../test/fromSkeletonLearning/noisy_bitore_skel.obj"
    
    process_obj_with_noise(input_file, output_file, 
                          noise_type='gaussian', 
                          std_dev=0.01)
