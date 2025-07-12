#!/bin/bash

# Configuration - Edit this array to define your files
SHAPEPATHS=(
    "normalized_points_file1.npz"
    "normalized_points_file2.npz"
    "normalized_points_file3.npz"
)

# Parameters - Edit these values as needed
CONFIG="conf.conf"
N_SURFACE=1000
N_QUERIES=10000
N_MINIMAX=1000
LAMDA_MAX=10
SIGMA=0.01

# Generate experiment directories based on shapepath + current date
CURRENT_DATE=$(date +"%d%m%Y")
EXP_DIRS=()

for shapepath in "${SHAPEPATHS[@]}"; do
    # Extract filename without extension
    filename=$(basename "$shapepath" .npz)
    # Create experiment directory name: filename_DDMMYYYY
    exp_dir="experiments/${filename}_${CURRENT_DATE}"
    EXP_DIRS+=("$exp_dir")
done

echo "Starting batch training and evaluation..."
echo "Files to process: ${#SHAPEPATHS[@]}"
echo "Config: $CONFIG"
echo "Parameters: n_surface=$N_SURFACE, n_queries=$N_QUERIES, n_minimax=$N_MINIMAX, lamda_max=$LAMDA_MAX, sigma=$SIGMA"
echo "----------------------------------------"

# Process each file
for i in "${!SHAPEPATHS[@]}"; do
    SHAPEPATH="${SHAPEPATHS[$i]}"
    EXP_DIR="${EXP_DIRS[$i]}"
    
    echo "[$((i+1))/${#SHAPEPATHS[@]}] Processing: $SHAPEPATH -> $EXP_DIR"
    
    # Create experiment directory
    mkdir -p "$EXP_DIR"
    
    # Training
    echo "  Training..."
    PYTORCH_ENABLE_MPS_FALLBACK=1 python train_socc.py \
        --device mps \
        --n_points 10000 \
        --name "$(basename "$SHAPEPATH" .npz)" \
        --n_surface "$N_SURFACE" \
        --lamda_max "$LAMDA_MAX" \
        --n_queries "$N_QUERIES" \
        --n_minimax "$N_MINIMAX" \
        --shapepath "$SHAPEPATH" \
        --exp_dir "$EXP_DIR" \
        --config "$CONFIG" \
        --sigma "$SIGMA"
    
    if [ $? -ne 0 ]; then
        echo "  ERROR: Training failed for $SHAPEPATH"
        continue
    fi
    
    # Evaluation
    echo "  Evaluating..."
    PYTORCH_ENABLE_MPS_FALLBACK=1 python eval.py \
        --device 0 \
        --shapename "$SHAPEPATH" \
        --results_dir "$EXP_DIR/" \
        --config "$CONFIG"
    
    if [ $? -ne 0 ]; then
        echo "  ERROR: Evaluation failed for $SHAPEPATH"
        continue
    fi
    
    echo "  ✓ Completed: $SHAPEPATH"
    echo "----------------------------------------"
done

echo "Batch processing complete!"
echo "Results saved in respective experiment directories."