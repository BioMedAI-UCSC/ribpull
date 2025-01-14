import sys

def normalize_vertices(vertices):
    """Normalize vertex coordinates to range [-1, 1] while preserving aspect ratio"""
    if not vertices:
        return vertices
    
    # Find min and max values for each dimension
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    z_coords = [v[2] for v in vertices]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    z_min, z_max = min(z_coords), max(z_coords)
    
    # Find the largest range to maintain aspect ratio
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = max(x_range, y_range, z_range)
    
    if max_range == 0:
        return vertices
    
    # Normalize vertices
    normalized = []
    for vertex in vertices:
        norm_x = 2 * (vertex[0] - x_min) / max_range - 1
        norm_y = 2 * (vertex[1] - y_min) / max_range - 1
        norm_z = 2 * (vertex[2] - z_min) / max_range - 1
        normalized.append([norm_x, norm_y, norm_z])
    
    return normalized

def convert_3d_file(input_path, output_path, normalize=True):
    vertices = []
    normals = []
    has_normals = False
    
    # Detect file type from extension
    file_type = input_path.split('.')[-1].lower()
    
    # According to the file type (ply, obj, xyz), read vertices and normals
    with open(input_path, 'r') as f:
        if file_type == 'obj':
            for line in f:
                if line.startswith('v '):  # vertex line
                    coords = line.strip().split()[1:]  # Get all values after 'v'
                    if len(coords) == 6:  # If we have both position and normal
                        vertices.append([float(x) for x in coords[0:3]])  # First 3 values are position
                        normals.append([float(x) for x in coords[3:6]])  # Last 3 values are normal
                        has_normals = True
        elif file_type == 'ply':
            # Skip header until 'end_header'
            header = True
            vertex_count = 0
            current_vertex = 0
            while header:
                line = f.readline().strip()
                if 'element vertex' in line:
                    vertex_count = int(line.split()[-1])
                elif 'property float nx' in line:
                    has_normals = True
                elif 'end_header' in line:
                    header = False
            # Read vertices and normals if present
            while current_vertex < vertex_count:
                line = f.readline().strip().split()
                vertices.append([float(x) for x in line[0:3]])
                if has_normals:
                    normals.append([float(x) for x in line[3:6]])
                current_vertex += 1
        elif file_type == 'xyz':
            for line in f:
                values = line.strip().split()
                if len(values) >= 6:  # Has normals
                    vertices.append([float(x) for x in values[0:3]])
                    normals.append([float(x) for x in values[3:6]])
                    has_normals = True
                elif len(values) >= 3:  # Only vertices
                    vertices.append([float(x) for x in values[0:3]])
    
    # Normalize vertices if requested
    if normalize:
        vertices = normalize_vertices(vertices)
        print("Vertices normalized to range [-1, 1]")
    
    # Write vertices to output file in XYZ format
    with open(output_path, 'w') as f:
        if has_normals and len(vertices) == len(normals):
            print(f"Converting {len(vertices)} vertices with normals...")
            for v, n in zip(vertices, normals):
                f.write(f"{v[0]} {v[1]} {v[2]} {n[0]} {n[1]} {n[2]}\n")
        else:
            print(f"Converting {len(vertices)} vertices without normals...")
            for v in vertices:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")

    print(f"Conversion complete: {input_path} → {output_path}")
    print(f"Total vertices: {len(vertices)}")
    if has_normals:
        print("Normals were included in the conversion")
    else:
        print("No normals were found in the input file")

def print_usage():
    print("Usage: python script.py input_file output_file [--no-normalize]")
    print("Supported input formats: .obj, .ply (ASCII), .xyz")
    print("Output will be in .xyz format")
    print("Example: python script.py model.obj output.xyz")
    print("Use --no-normalize flag to keep original coordinates")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    normalize = True
    
    # Check for --no-normalize flag
    if len(sys.argv) > 3 and sys.argv[3] == '--no-normalize':
        normalize = False
    
    # Check if input file extension is supported
    supported_formats = ['obj', 'ply', 'xyz']
    input_format = input_file.split('.')[-1].lower()
    
    if input_format not in supported_formats:
        print(f"Error: Unsupported input format '.{input_format}'")
        print(f"Supported formats: {', '.join(supported_formats)}")
        sys.exit(1)
    
    try:
        convert_3d_file(input_file, output_file, normalize)
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        sys.exit(1)
        