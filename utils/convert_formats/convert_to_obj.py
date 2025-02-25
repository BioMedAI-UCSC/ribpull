#!/usr/bin/env python3

def xyz_to_obj(input_file, output_file):
    """
    Convert XYZ point cloud data to OBJ format.
    
    Args:
        input_file (str): Path to input XYZ file
        output_file (str): Path to output OBJ file
    """
    vertices = []
    
    # Read XYZ file
    with open(input_file, 'r') as f:
        for line in f:
            # Skip empty lines and comments
            if not line.strip() or line.startswith('#'):
                continue
            
            # Parse x, y, z coordinates
            coords = line.strip().split()
            if len(coords) >= 3:
                x, y, z = map(float, coords[:3])
                vertices.append((x, y, z))
    
    # Write OBJ file
    with open(output_file, 'w') as f:
        # Write header comment
        f.write("# Converted from XYZ format\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write point cloud as vertices only
        f.write("# Point cloud - each vertex represents a point\n")
        for i in range(1, len(vertices) + 1):
            f.write(f"p {i}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert XYZ point cloud to OBJ format')
    parser.add_argument('input', help='Input XYZ file')
    parser.add_argument('output', help='Output OBJ file')
    
    args = parser.parse_args()
    
    try:
        xyz_to_obj(args.input, args.output)
        print(f"Successfully converted {args.input} to {args.output}")
    except Exception as e:
        print(f"Error: {str(e)}")