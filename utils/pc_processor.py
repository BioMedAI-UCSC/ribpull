import numpy as np
import json
from pathlib import Path
import trimesh

def load_and_preprocess(obj_path):
    """
    Load an OBJ file and return vertices
    """
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                coords = line.split()[1:]
                vertices.append([float(x) for x in coords])
    raw_vertices = np.array(vertices)
    return raw_vertices

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

def load_skeleton_obj(file_path):
    mesh = trimesh.load_mesh(file_path)
    return np.array(mesh.vertices)

def load_skeleton_xyz(file_path): #assumes space-separated X, Y, Z
    return np.loadtxt(file_path, delimiter=' ')

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

def create_visualization_obj(points, edges, endpoint_info, output_file, marker_radius=None, blend_colors=False):
    """Modified to handle endpoints detected by multiple methods."""
    if marker_radius is None:
        if len(edges) > 0:
            edge_lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
            marker_radius = np.mean(edge_lengths) * 2
        else:
            bbox_size = np.ptp(points, axis=0)
            marker_radius = np.mean(bbox_size) * 0.02

    colors = {
        'connectivity': [255, 0, 0],    # Red
        'density': [0, 255, 0],         # Green
        'edge_similarity': [0, 0, 255]  # Blue
    }

    # Priority order for methods (if not blending)
    method_priority = ['connectivity', 'density', 'edge_similarity']

    all_vertices = points.tolist()
    all_vertex_colors = [[128, 128, 128] for _ in range(len(points))]  # Default gray
    all_faces = []
    vertex_offset = len(all_vertices)

    # First, collect all methods for each endpoint
    endpoint_methods = {}
    for component, endpoints_data in endpoint_info.items():
        for endpoint, method in endpoints_data:
            if endpoint not in endpoint_methods:
                endpoint_methods[endpoint] = set()
            endpoint_methods[endpoint].add(method)

    # Process each endpoint
    for component, endpoints_data in endpoint_info.items():
        for endpoint, _ in endpoints_data:
            if endpoint in endpoint_methods:  # Check if we haven't processed this endpoint yet
                methods = endpoint_methods[endpoint]
                
                if blend_colors and len(methods) > 1:
                    # Blend colors of all methods that detected this endpoint
                    color = [0, 0, 0]
                    for method in methods:
                        method_color = colors[method]
                        color = [c1 + c2 for c1, c2 in zip(color, method_color)]
                    # Average the colors
                    color = [min(255, c // len(methods)) for c in color]
                else:
                    # Use the highest priority method's color
                    for method in method_priority:
                        if method in methods:
                            color = colors[method]
                            break
                    else:
                        color = colors[list(methods)[0]]  # Fallback to first method if none in priority

                sphere_verts, sphere_faces = create_sphere_marker(
                    points[endpoint], 
                    marker_radius
                )
                
                # Add vertices and their colors
                all_vertices.extend(sphere_verts.tolist())
                all_vertex_colors.extend([color for _ in range(len(sphere_verts))])
                
                all_faces.extend((sphere_faces + vertex_offset).tolist())
                vertex_offset += len(sphere_verts)
                
                # Remove this endpoint from our tracking dict to avoid processing it again
                del endpoint_methods[endpoint]

    # Write PLY file
    with open(output_file, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(all_vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(all_faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")

        # Vertices with colors
        for vertex, color in zip(all_vertices, all_vertex_colors):
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]} {color[0]} {color[1]} {color[2]}\n")
        
        # Faces
        for face in all_faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

        # Edges
        for edge in edges:
            f.write(f"{edge[0]} {edge[1]}\n")

def load_skeleton_file(file_path):
    """Load skeleton points from either OBJ or XYZ."""
    extension = Path(file_path).suffix.lower()
    if extension == ".obj":
        return load_skeleton_obj(file_path)
    elif extension == ".xyz":
        return load_skeleton_xyz(file_path)
    else:
        raise ValueError(f"Unsupported file format: {extension}")

def numpy_to_python(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    return obj

# When you need to serialize your results:
def save_results(fracture_results, output_file=None):
    # Convert all numpy types to Python types
    serializable_results = numpy_to_python(fracture_results)
    
    # Print or save to file
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    else:
        print(json.dumps(serializable_results, indent=2))

def save_obj_with_markers(file_path, vertices, discontinuity_locations):
    """
    Save vertices to OBJ with additional marker objects at discontinuity points.
    
    Args:
        file_path (str): Output path for the OBJ file
        vertices (numpy.ndarray): Array of vertex coordinates with shape (N, 3)
        discontinuity_locations (list): Indices of vertices to highlight
    """
    base_name = file_path.rsplit('.', 1)[0]
    
    # Save original points
    with open(f"{base_name}.obj", 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f'v {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n')
    
    # Save markers as a separate OBJ
    with open(f"{base_name}_markers.obj", 'w') as f:
        # Create small cubes at discontinuity points
        scale = 0.02  # Size of marker
        
        vertex_count = 0
        for idx in discontinuity_locations:
            point = vertices[idx]
            # Define cube vertices around the point
            cube_verts = [
                [point[0] - scale, point[1] - scale, point[2] - scale],
                [point[0] + scale, point[1] - scale, point[2] - scale],
                [point[0] + scale, point[1] + scale, point[2] - scale],
                [point[0] - scale, point[1] + scale, point[2] - scale],
                [point[0] - scale, point[1] - scale, point[2] + scale],
                [point[0] + scale, point[1] - scale, point[2] + scale],
                [point[0] + scale, point[1] + scale, point[2] + scale],
                [point[0] - scale, point[1] + scale, point[2] + scale],
            ]
            
            # Write cube vertices
            for v in cube_verts:
                f.write(f'v {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n')
            
            # Write cube faces
            faces = [
                [1,2,3,4], [5,6,7,8],  # top, bottom
                [1,2,6,5], [2,3,7,6],  # sides
                [3,4,8,7], [4,1,5,8]   # sides
            ]
            
            # Adjust indices for current cube
            base = vertex_count * 8
            for face in faces:
                adjusted = [i + base for i in face]
                f.write(f'f {adjusted[0]} {adjusted[1]} {adjusted[2]} {adjusted[3]}\n')
            
            vertex_count += 1