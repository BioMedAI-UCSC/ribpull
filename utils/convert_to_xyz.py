import sys

def convert_3d_file(input_path, output_path):
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
    print("Usage: python script.py input_file output_file")
    print("Supported input formats: .obj, .ply (ASCII), .xyz")
    print("Output will be in .xyz format")
    print("Example: python script.py model.obj output.xyz")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print_usage()
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Check if input file extension is supported
    supported_formats = ['obj', 'ply', 'xyz']
    input_format = input_file.split('.')[-1].lower()
    
    if input_format not in supported_formats:
        print(f"Error: Unsupported input format '.{input_format}'")
        print(f"Supported formats: {', '.join(supported_formats)}")
        sys.exit(1)
    
    try:
        convert_3d_file(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        sys.exit(1)
        