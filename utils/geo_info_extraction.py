import numpy as np
from scipy.spatial import Delaunay
import argparse
import json
import sys
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
import trimesh
from sklearn.neighbors import NearestNeighbors
from ripser import ripser
from persim import plot_diagrams

def analyze_obj_file(file_path):
    """
    Perform geometric analysis on a 3D mesh represented by an OBJ file.
    
    Parameters:
    file_path (str): Path to the OBJ file
    
    Returns:
    dict: A dictionary containing various geometric properties of the mesh
    """
    # Load the OBJ file
    vertices, faces = load_obj_file(file_path)
    
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
    
    # Compute surface area and volume
    surface_area = compute_surface_area(vertices, faces)
    volume = compute_volume(vertices, faces)
    
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
            "Number of Edges": num_edges,
            "Number of Faces": num_faces,
            "Surface Area": f"{surface_area:.6f}",
            "Volume": f"{volume:.6f}"
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

def load_obj_file(file_path):
    """
    Load vertex and face data from an OBJ file.
    
    Parameters:
    file_path (str): Path to the OBJ file
    
    Returns:
    np.ndarray, np.ndarray: Vertex coordinates, face indices
    """
    vertices = []
    faces = []
    
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                vertex = [float(x) for x in line.split()[1:4]]
                vertices.append(vertex)
            elif line.startswith("f "):
                face = [int(x.split("/")[0]) - 1 for x in line.split()[1:4]]
                faces.append(face)
    
    return np.array(vertices), np.array(faces)

def compute_surface_area(vertices, faces):
    """
    Compute the surface area of a 3D mesh.
    
    Parameters:
    vertices (np.ndarray): Vertex coordinates
    faces (np.ndarray): Face indices
    
    Returns:
    float: Surface area of the mesh
    """
    surface_area = 0
    for face in faces:
        p1 = vertices[face[0]]
        p2 = vertices[face[1]]
        p3 = vertices[face[2]]
        surface_area += 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
    return surface_area

def compute_volume(vertices, faces):
    """
    Compute the volume of a 3D mesh.
    
    Parameters:
    vertices (np.ndarray): Vertex coordinates
    faces (np.ndarray): Face indices
    
    Returns:
    float: Volume of the mesh
    """
    volume = 0
    for face in faces:
        p1 = vertices[face[0]]
        p2 = vertices[face[1]]
        p3 = vertices[face[2]]
        volume += p1[0] * (p2[1] * p3[2] - p3[1] * p2[2])
        volume -= p1[1] * (p2[0] * p3[2] - p3[0] * p2[2])
        volume += p1[2] * (p2[0] * p3[1] - p3[0] * p2[1])
    return abs(volume) / 6

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

def load_and_preprocess(obj_path):
    mesh = trimesh.load_mesh(obj_path)
    points = mesh.vertices
    return points

def compute_normals(points, k=10):
    # Compute normals using PCA on local neighborhoods
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

def detect_discontinuities(normals, threshold=0.2):
    # Compute angles between adjacent normals
    angles = np.arccos(np.clip(np.sum(normals[:-1] * normals[1:], axis=1), -1.0, 1.0))
    discontinuities = np.where(angles > threshold)[0]
    return discontinuities

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

def compute_zscore_distance(points1, points2):
    mean1 = np.mean(points1, axis=0)
    std1 = np.std(points1, axis=0)
    zscore = np.abs((points2 - mean1) / std1)
    return np.mean(zscore)

def compute_betti_numbers(points, max_dim=1):
    # Compute persistent homology
    diagrams = ripser(points, maxdim=max_dim)['dgms']
    betti_numbers = [len(diagram) for diagram in diagrams]
    return betti_numbers

def analyze_fractures(subject_path, reference_path):
    # Load point clouds
    subject_points = load_and_preprocess(subject_path)
    reference_points = load_and_preprocess(reference_path)
    
    # Compute normals and detect discontinuities
    subject_normals = compute_normals(subject_points)
    reference_normals = compute_normals(reference_points)
    
    discontinuities = detect_discontinuities(subject_normals)
    
    # Compute distances
    chamfer_dist = compute_chamfer_distance(subject_points, reference_points)
    mahalanobis_dist = compute_mahalanobis_distance(subject_points, reference_points)
    zscore_dist = compute_zscore_distance(subject_points, reference_points)
    
    # Compute Betti numbers
    subject_betti = compute_betti_numbers(subject_points)
    reference_betti = compute_betti_numbers(reference_points)
    
    results = {
        'discontinuity_locations': discontinuities,
        'chamfer_distance': chamfer_dist,
        'mahalanobis_distance': mahalanobis_dist,
        'zscore_distance': zscore_dist,
        'subject_betti': subject_betti,
        'reference_betti': reference_betti
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Perform geometric analysis on an OBJ file")
    parser.add_argument("file_path", help="Path to the OBJ file")
    parser.add_argument("reference_path", help="Path to the reference OBJ file", default=None, nargs='?')
    args = parser.parse_args()

    input_file = args.file_path
    
    results = analyze_obj_file(input_file)
    print(json.dumps(results, indent=2))
    if len(sys.argv) > 2:
        reference_file = args.reference_path
        fracture_results = analyze_fractures(input_file, reference_file)
        print(json.dumps(fracture_results, indent=2))
        
        

import time
start_time = time.time()

if __name__ == "__main__":
    main()

print("--- %s seconds ---" % (time.time() - start_time))