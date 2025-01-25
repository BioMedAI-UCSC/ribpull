#!/bin/bash

# Set base directories
INPUT_DIR="test/inference_test_streamline"
VISUALIZATION_PLY_DIR="test/visualize_test_streamline"
XYZ_NORMAL_DIR="test/xyz_test_streamline"
OBJ_OUTPUT_DIR="test/obj_test_streamline"

# Ensure output directories exist
mkdir -p "$VISUALIZATION_PLY_DIR"
mkdir -p "$XYZ_NORMAL_DIR"
mkdir -p "$OBJ_OUTPUT_DIR"


echo "Splitting foreground and background from RibSeg's output..."

# Step 1: Split foreground and background
python utils/split_fgnd_bgnd.py "$INPUT_DIR" "$VISUALIZATION_PLY_DIR"

echo "Converting PLY files to normalized xyz with calculated normals..."

# Step 2: Convert to XYZ with normals
python utils/convert_to_xyz.py "$VISUALIZATION_PLY_DIR" "$XYZ_NORMAL_DIR"

echo "For each XYZ file, run Neural Skeleton to extract skeleton as OBJ file..."

# Iterate through all XYZ files in the XYZ_NORMAL_DIR
for xyz_file in "$XYZ_NORMAL_DIR"/*.xyz; do
    # Check if the file exists
    if [ -f "$xyz_file" ]; then
        # Generate output filename (replace .xyz with .obj)
        output_obj="$OBJ_OUTPUT_DIR/$(basename "${xyz_file%.*}").obj"
        
        # Run main.py to get the skeleton of the XYZ file in OBJ format
        python main.py "$xyz_file" "$output_obj"
    fi
done

echo "Run endpoint detector..."

# Step 4: Run endpoint detector on all generated OBJ files to generate OBJ with markers showing detected fractures
python utils/endpoint_detector.py "$OBJ_OUTPUT_DIR"

echo "Fracture detection complete!"