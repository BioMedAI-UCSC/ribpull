from scipy.spatial import Delaunay, cKDTree
import argparse
import sys
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize
import pc_processor

def analyze_obj_file(file_path):
    """
    Perform geometric analysis on a 3D mesh represented by an OBJ file.
    
    Parameters:
    file_path (str): Path to the OBJ file
    
    Returns:
    dict: A dictionary containing various geometric properties of the mesh
    """
    # Load the OBJ file
    vertices, faces = pc_processor.load_obj_file(file_path)
    
    # Compute basic properties
    num_vertices = len(vertices)
    num_faces = len(faces)
    
    # Compute edges (needed for correct Euler characteristic)
    edges = set()
    for face in faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edges.add(edge)
    num_edges = len(edges)
    
    # Compute Delaunay triangulation for additional analysis
    tri = Delaunay(vertices)
    edge_lengths = compute_edge_lengths(vertices, tri.simplices)
    
    if edge_lengths.size:
        avg_edge_length = np.mean(edge_lengths)
        max_edge_length = np.max(edge_lengths)
    else:
        avg_edge_length = 0
        max_edge_length = 0
    
    # Compute correct topological properties
    euler_characteristic = num_vertices - num_edges + num_faces
    genus = (2 - euler_characteristic) // 2
    
    # Construct the result dictionary
    result = {
        "Mesh Statistics": {
            "Number of Vertices": num_vertices,
            "Number of Edges": num_edges
            },
        "Edge Analysis": {
            "Average Edge Length": f"{avg_edge_length:.6f}",
            "Maximum Edge Length": f"{max_edge_length:.6f}"
        },
        "Topological Properties": {
            "Euler Characteristic": euler_characteristic,
            "Genus (number of holes)": genus
        }
    }
    return result

def compute_edge_lengths(vertices, simplices):
    """
    Compute the lengths of the edges in a Delaunay triangulation.
    
    Parameters:
    vertices (np.ndarray): Vertex coordinates
    simplices (np.ndarray): Delaunay simplices (triangle indices)
    
    Returns:
    np.ndarray: Edge lengths
    """
    if not simplices.size:
        return np.array([])
    
    edge_lengths = []
    for simplex in simplices:
        for i in range(3):
            p1 = vertices[simplex[i]]
            p2 = vertices[simplex[(i + 1) % 3]]
            edge_lengths.append(np.linalg.norm(p2 - p1))
    return np.array(edge_lengths)

def compute_chamfer_distance(points1, points2):
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    dist1, _ = tree1.query(points2)
    dist2, _ = tree2.query(points1)
    
    chamfer_dist = np.mean(dist1) + np.mean(dist2)
    return chamfer_dist

def compute_mahalanobis_distance(points1, points2):
    mean1 = np.mean(points1, axis=0)
    cov1 = np.cov(points1.T)
    inv_cov = np.linalg.inv(cov1)
    
    diff = points2 - mean1
    mahalanobis_dist = np.sqrt(np.sum(np.dot(diff, inv_cov) * diff, axis=1))
    return np.mean(mahalanobis_dist)

def fit_sphere(points, initial_radius=1.0):
    """
    Fit a sphere to a set of 3D points using nonlinear optimization.
    
    Args:
        points (np.ndarray): Nx3 array of point coordinates
        initial_radius (float): Initial guess for sphere radius
        
    Returns:
        tuple: (center coordinates, radius, fit error)
    """
    def objective(params):
        center = params[:3]
        radius = params[3]
        distances = np.sqrt(np.sum((points - center)**2, axis=1))
        return np.mean((distances - radius)**2)
    
    initial_guess = np.concatenate([np.mean(points, axis=0), [initial_radius]])
    result = minimize(objective, initial_guess, method='Nelder-Mead')
    
    return result.x[:3], result.x[3], result.fun

def detect_fractures_sphere(points, k_neighbors=10, error_threshold=0.1):
    """
    Detect fractures using sphere regression method.
    """
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    sphere_errors = np.zeros(len(points))
    
    for i in range(len(points)):
        neighbor_points = points[indices[i]]
        _, _, error = fit_sphere(neighbor_points)
        sphere_errors[i] = error
    
    # Points with high sphere fitting error are likely fracture points
    fractures = np.where(sphere_errors > error_threshold)[0]
    return fractures, sphere_errors

def compute_normals(points, k=10):
    """Existing normal computation function"""
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    normals = []
    for idx in indices:
        local_points = points[idx]
        centroid = np.mean(local_points, axis=0)
        local_centered = local_points - centroid
        cov = np.cov(local_centered.T)
        eigenvals, eigenvecs = np.linalg.eig(cov)
        normal = eigenvecs[:, np.argmin(eigenvals)]
        normals.append(normal)
    
    return np.array(normals)

def detect_discontinuities(points, normals, k=4, threshold=1.5):
    """Existing normal-based discontinuity detection"""
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, indices = nbrs.kneighbors(points)
    
    discontinuities = []
    for i, neighbors in enumerate(indices):
        neighbor_angles = np.arccos(np.clip(
            np.dot(normals[neighbors], normals[i]), -1.0, 1.0))
        if np.mean(neighbor_angles) > threshold:
            discontinuities.append(i)
            
    return np.array(discontinuities)

def analyze_fractures(subject_path, reference_path, method='normals', sphere_params=None):
    """
    Enhanced analyze_fractures function with multiple detection methods.
    
    Args:
        subject_path: Path to subject point cloud
        reference_path: Path to reference point cloud
        method: 'normals' or 'sphere' for detection method
        sphere_params: Dictionary with 'k_neighbors' and 'error_threshold' for sphere method
    """
    # Load point clouds
    subject_points = pc_processor.load_and_preprocess(subject_path)
    reference_points = pc_processor.load_and_preprocess(reference_path)
    
    # Default sphere parameters
    if sphere_params is None:
        sphere_params = {
            'k_neighbors': 10,
            'error_threshold': 0.1
        }
    
    # Detect fractures using selected method
    if method == 'sphere':
        discontinuities, sphere_errors = detect_fractures_sphere(
            subject_points,
            k_neighbors=sphere_params['k_neighbors'],
            error_threshold=sphere_params['error_threshold']
        )
        method_specific_data = {'sphere_errors': sphere_errors}
    else:  # method == 'normals'
        subject_normals = compute_normals(subject_points)
        discontinuities = detect_discontinuities(subject_points, subject_normals)
        method_specific_data = {'normals': subject_normals}
    
    # Compute distances
    chamfer_dist = compute_chamfer_distance(subject_points, reference_points)
    mahalanobis_dist = compute_mahalanobis_distance(subject_points, reference_points)
    
    results = {
        'method': method,
        'discontinuity_locations': discontinuities,
        'discontinuity_count': len(discontinuities),
        'chamfer_distance': chamfer_dist,
        'mahalanobis_distance': mahalanobis_dist,
        'method_specific_data': method_specific_data
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Perform geometric analysis on an OBJ file")
    parser.add_argument("file_path", help="Path to the OBJ file")
    parser.add_argument("reference_path", help="Path to the reference OBJ file", default=None, nargs='?')
    parser.add_argument("--method", choices=['normals', 'sphere'], default='sphere',
                      help="Method to use for fracture detection")
    parser.add_argument("--sphere-k", type=int, default=3,
                      help="Number of neighbors for sphere fitting")
    parser.add_argument("--sphere-threshold", type=float, default=0.0001,
                      help="Error threshold for sphere fitting")
    
    args = parser.parse_args()

    input_file = args.file_path
    
    results = analyze_obj_file(input_file)
    pc_processor.save_results(results, output_file=None)
    
    if args.reference_path:
        sphere_params = {
            'k_neighbors': args.sphere_k,
            'error_threshold': args.sphere_threshold
        }
        
        fracture_results = analyze_fractures(
            input_file,
            args.reference_path,
            method=args.method,
            sphere_params=sphere_params
        )
        
        pc_processor.save_results(fracture_results, output_file=None)
        pc_processor.save_obj_with_markers(
            input_file,
            pc_processor.load_and_preprocess(input_file),
            fracture_results['discontinuity_locations']
        )

import time
start_time = time.time()

if __name__ == "__main__":
    main()

print("--- %s seconds ---" % (time.time() - start_time))