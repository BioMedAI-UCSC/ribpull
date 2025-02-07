import numpy as np
from scipy.spatial import cKDTree
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import argparse
from pathlib import Path

def load_skeleton_obj(file_path):
    mesh = trimesh.load_mesh(file_path)
    return np.array(mesh.vertices)

def load_skeleton_xyz(file_path): #assumes space-separated X, Y, Z
    return np.loadtxt(file_path, delimiter=' ')

def load_skeleton_file(file_path):
    """Load skeleton points from either OBJ or XYZ."""
    extension = Path(file_path).suffix.lower()
    if extension == ".obj":
        return load_skeleton_obj(file_path)
    elif extension == ".xyz":
        return load_skeleton_xyz(file_path)
    else:
        raise ValueError(f"Unsupported file format: {extension}")
    
def create_sphere_marker(center, radius, resolution=10):
    """Create vertices and faces for a sphere marker."""
    # Create a unit sphere
    phi = np.linspace(0, 2*np.pi, resolution)
    theta = np.linspace(0, np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)

    # Convert to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi) + center[0]
    y = radius * np.sin(theta) * np.sin(phi) + center[1]
    z = radius * np.cos(theta) + center[2]

    # Create vertices & faces
    vertices = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    faces = []
    for i in range(resolution-1):
        for j in range(resolution-1):
            v0 = i * resolution + j
            v1 = i * resolution + (j + 1)
            v2 = (i + 1) * resolution + j
            v3 = (i + 1) * resolution + (j + 1)
            
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    return vertices, np.array(faces)

def detect_by_edge_similarity(points, edges):
    """
    Detect endpoints by finding points with small angles between edges (indicating
    edges in similar directions) versus points with angles close to 180° (indicating
    opposite directions).
    """
    endpoint_mask = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):
        connected_edges = edges[np.any(edges == i, axis=1)]
        if len(connected_edges) < 2:
            continue
        connected_verts = np.unique(connected_edges[connected_edges != i])
        edge_vectors = points[connected_verts] - points[i]
        edge_vectors = edge_vectors / np.linalg.norm(edge_vectors, axis=1)[:, np.newaxis]
        # Check if all angles are less than 36 degrees (cos > 0.809)
        is_endpoint = True
        for j in range(len(edge_vectors)):
            for k in range(j+1, len(edge_vectors)):
                cos_angle = abs(np.dot(edge_vectors[j], edge_vectors[k]))
                if cos_angle < np.cos(np.radians(18)):  # angle > 18 degrees
                    is_endpoint = False
                    break
            if not is_endpoint:
                break    
        endpoint_mask[i] = is_endpoint
    return endpoint_mask

def construct_radius_connectivity(points, radius=None):
    """Construct connectivity graph using radius-based neighbor search."""
    if radius is None:
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=3)
        radius = np.mean(distances[:, 1]) * 2.45
    print (f"Using radius: {radius}")
    
    tree = cKDTree(points)
    pairs = list(tree.query_pairs(radius))
    edges = np.array(pairs) if pairs else np.zeros((0, 2), dtype=int)
    
    return edges, radius

def detect_endpoints_by_density(points, box_size=None):
    """Detect endpoints by counting points within a local box neighborhood."""
    if box_size is None:
        # Compute box size based on point cloud characteristics
        bbox = np.ptp(points, axis=0)  # Get range of points in each dimension
        box_size = np.mean(bbox) * 0.047  # Use 10% of mean bbox size
    
    tree = cKDTree(points)
    endpoint_mask = np.zeros(len(points), dtype=bool)
    
    # For each point, count neighbors within box_size
    for i in range(len(points)):
        query_point = points[i]
        
        # Define box bounds
        box_min = query_point - box_size/2
        box_max = query_point + box_size/2
        
        # Find points within box
        indices = tree.query_ball_point(query_point, box_size/2)
        n_neighbors = len(indices)
        
        # If fewer than threshold neighbors, mark as endpoint
        if n_neighbors < 10:  # You can adjust this threshold
            endpoint_mask[i] = True
            
    return endpoint_mask

def identify_separate_skeletons(points, edges, n_points):
    """Identify separate skeleton components in the point cloud."""
    adj_matrix = csr_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(n_points, n_points)
    )
    adj_matrix = adj_matrix + adj_matrix.T
    
    n_components, labels = connected_components(
        csgraph=adj_matrix,
        directed=False,
        return_labels=True
    )
    
    return n_components, labels

def find_endpoints_per_skeleton(edges, n_points, component_labels):
    """Find endpoints for each separate skeleton component."""
    connections = np.zeros(n_points)
    for edge in edges:
        connections[edge[0]] += 1
        connections[edge[1]] += 1
    
    component_endpoints = {}
    unique_components = np.unique(component_labels)
    
    for component in unique_components:
        component_mask = (component_labels == component)
        component_connections = connections[component_mask]
        local_endpoints = np.where(component_connections == 1)[0]
        global_endpoints = np.where(component_mask)[0][local_endpoints]
        component_endpoints[component] = global_endpoints
    
    return component_endpoints, connections

def create_visualization_obj(points, edges, component_endpoints, output_file, marker_radius=None):
    """Create OBJ file with skeleton and endpoint markers."""
    if marker_radius is None:
        if len(edges) > 0:
            edge_lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
            marker_radius = np.mean(edge_lengths) * 2
        else:
            bbox_size = np.ptp(points, axis=0)
            marker_radius = np.mean(bbox_size) * 0.02

    all_vertices = points.tolist()
    all_faces = []
    vertex_offset = len(all_vertices)

    for component, endpoints in component_endpoints.items():
        for endpoint in endpoints:
            sphere_verts, sphere_faces = create_sphere_marker(
                points[endpoint], 
                marker_radius
            )
            
            all_vertices.extend(sphere_verts.tolist())
            all_faces.extend((sphere_faces + vertex_offset).tolist())
            vertex_offset += len(sphere_verts)

    with open(output_file, 'w') as f:
        for v in all_vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        for edge in edges:
            f.write(f"l {edge[0]+1} {edge[1]+1}\n")
        
        for face in all_faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def analyze_and_visualize_skeleton(input_file, output_obj, radius=None):
    points = load_skeleton_file(input_file)
    edges, used_radius = construct_radius_connectivity(points, radius)
    
    # Get component information
    n_components, component_labels = identify_separate_skeletons(
        points, edges, len(points)
    )
    
    # Get endpoints from all three methods
    connectivity_endpoints, connections = find_endpoints_per_skeleton(
        edges, len(points), component_labels
    )
    density_endpoints = detect_endpoints_by_density(points)
    edge_similarity_endpoints = detect_by_edge_similarity(points, edges)
    
    # Combine all three methods while preserving component structure
    combined_endpoints = {}
    for component in range(n_components):
        component_mask = (component_labels == component)
        component_points = np.where(component_mask)[0]
        
        # Find endpoints using all three methods
        connectivity_ends = set(connectivity_endpoints.get(component, []))
        density_ends = set(component_points[density_endpoints[component_mask]])
        similarity_ends = set(component_points[edge_similarity_endpoints[component_mask]])
        
        # Combine endpoints for this component
        combined_endpoints[component] = np.array(list(
            connectivity_ends.union(density_ends).union(similarity_ends)
        ))
    
    create_visualization_obj(points, edges, combined_endpoints, output_obj)
    
    return n_components, combined_endpoints

def process_directory(directory_path, radius=None):
    """Process all OBJ and XYZ files in the given directory."""
    directory = Path(directory_path)
    results = {}
    
    # Create output directory for results
    output_dir = directory / "endpoint_analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Find all OBJ and XYZ files
    skeleton_files = list(directory.glob("*.obj")) + list(directory.glob("*.xyz"))
    if not skeleton_files:
        print(f"No OBJ or XYZ files found in {directory}")
        return
    
    print(f"Found {len(skeleton_files)} files to process")
    
    # Process each file
    for file in skeleton_files:
        # Skip files that already have _endpoints suffix
        if file.stem.endswith("_endpoints"):
            continue
        print(f"\nProcessing: {file.name}")
        # Create output path (always save as OBJ)
        output_file = output_dir / f"{file.stem}_endpoints.obj"
        
        try:
            # Process the file
            n_components, component_endpoints = analyze_and_visualize_skeleton(
                str(file), str(output_file), radius
            )
            
            # Store results
            results[file.name] = {
                'n_components': n_components,
                'endpoints_per_component': {comp: len(endpoints) 
                                         for comp, endpoints in component_endpoints.items()}
            }
            
            print(f"Found {n_components} separate skeletons")
            for comp, endpoints in component_endpoints.items():
                print(f"Skeleton {comp}: {len(endpoints)} endpoints")
            print(f"Visualization saved to: {output_file}")
            
        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")
            continue
    
    # Write summary report
    report_path = output_dir / "analysis_summary.txt"
    with open(report_path, 'w') as f:
        f.write("Skeleton Analysis Summary\n")
        f.write("=======================\n\n")
        for filename, data in results.items():
            f.write(f"File: {filename}\n")
            f.write(f"Number of skeletons: {data['n_components']}\n")
            f.write("Endpoints per skeleton:\n")
            for comp, n_endpoints in data['endpoints_per_component'].items():
                f.write(f"  Skeleton {comp}: {n_endpoints} endpoints\n")
            f.write("\n")
    
    print(f"\nAnalysis complete. Summary report saved to: {report_path}")
    
def main():
    parser = argparse.ArgumentParser(description="Perform geometric analysis on OBJ files in a directory")
    parser.add_argument("directory_path", help="Path to the directory containing OBJ files")
    parser.add_argument("--radius", type=float, help="Optional: Specify connectivity radius", default=None)
    args = parser.parse_args()
    process_directory(args.directory_path, args.radius)

if __name__ == "__main__":
    main()