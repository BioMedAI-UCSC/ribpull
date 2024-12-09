import numpy as np
import json

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