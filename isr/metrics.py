#!/usr/bin/env python3
"""
Script to calculate mean and standard deviation of Chamfer-L1 and Chamfer-L2 metrics
from evaluation_log.txt files in subdirectories.
"""

import os
import re
import numpy as np
from pathlib import Path

def parse_evaluation_log(file_path):
    """
    Parse evaluation_log.txt file and extract Chamfer metrics.
    
    Args:
        file_path (str): Path to the evaluation_log.txt file
        
    Returns:
        tuple: (chamfer_l1, chamfer_l2) or (None, None) if parsing fails
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
            
        # Pattern to match the log format
        # Example: "2025-07-17 11:30:09, RibFrac1-rib-seg_norm_, Chamfer-L1: 0.005401, Chamfer-L2: 0.000068, Normals: nan"
        pattern = r'Chamfer-L1:\s*([\d.-]+),\s*Chamfer-L2:\s*([\d.-]+)'
        match = re.search(pattern, content)
        
        if match:
            chamfer_l1 = float(match.group(1))
            chamfer_l2 = float(match.group(2))
            return chamfer_l1, chamfer_l2
        else:
            print(f"Warning: Could not parse metrics from {file_path}")
            return None, None
            
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return None, None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

def collect_chamfer_metrics(base_directory="./experiments/17072025_total/"):
    """
    Collect all Chamfer-L1 and Chamfer-L2 metrics from subdirectories.
    
    Args:
        base_directory (str): Base directory to search in
        
    Returns:
        tuple: (list of chamfer_l1 values, list of chamfer_l2 values)
    """
    chamfer_l1_values = []
    chamfer_l2_values = []
    
    base_path = Path(base_directory)
    
    # Find all subdirectories that match the RibFrac pattern
    subdirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('RibFrac')]
    
    print(f"Found {len(subdirs)} RibFrac subdirectories")
    
    for subdir in sorted(subdirs):
        log_file = subdir / "evaluation_log.txt"
        print(f"Processing: {subdir.name}")
        
        chamfer_l1, chamfer_l2 = parse_evaluation_log(log_file)
        
        if chamfer_l1 is not None and chamfer_l2 is not None:
            chamfer_l1_values.append(chamfer_l1)
            chamfer_l2_values.append(chamfer_l2)
            print(f"  ✓ Chamfer-L1: {chamfer_l1:.6f}, Chamfer-L2: {chamfer_l2:.6f}")
        else:
            print(f"  ✗ Failed to extract metrics")
    
    return chamfer_l1_values, chamfer_l2_values

def calculate_stats(values, metric_name, log_file):
    """
    Calculate and write statistics for a list of values to log file.
    
    Args:
        values (list): List of numeric values
        metric_name (str): Name of the metric for display
        log_file (file): Open file handle for writing
    """
    if not values:
        log_file.write(f"\n{metric_name}: No valid values found\n")
        print(f"{metric_name}: No valid values found")
        return None, None
    
    values_array = np.array(values)
    mean_val = np.mean(values_array)
    std_val = np.std(values_array, ddof=1)  # Sample standard deviation
    min_val = np.min(values_array)
    max_val = np.max(values_array)
    
    log_file.write(f"\n{metric_name} Statistics:\n")
    log_file.write(f"  Count: {len(values)}\n")
    log_file.write(f"  Mean: {mean_val:.8f}\n")
    log_file.write(f"  Std Dev: {std_val:.8f}\n")
    log_file.write(f"  Min: {min_val:.8f}\n")
    log_file.write(f"  Max: {max_val:.8f}\n")
    
    print(f"{metric_name}: Mean = {mean_val:.8f}, Std = {std_val:.8f}")
    
    return mean_val, std_val

def main():
    """Main function to run the analysis."""
    print("Chamfer Metrics Statistics Calculator")
    print("=" * 40)
    
    # Collect metrics from all subdirectories
    chamfer_l1_values, chamfer_l2_values = collect_chamfer_metrics()
    
    # Create log file with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open("log.txt", "w") as log_file:
        log_file.write("Chamfer Metrics Statistics Report\n")
        log_file.write("=" * 40 + "\n")
        log_file.write(f"Generated: {timestamp}\n")
        log_file.write(f"Total files processed: {len(chamfer_l1_values)}\n")
        
        # Calculate and write statistics to log file
        l1_mean, l1_std = calculate_stats(chamfer_l1_values, "Chamfer-L1", log_file)
        l2_mean, l2_std = calculate_stats(chamfer_l2_values, "Chamfer-L2", log_file)
        
        # Write summary to log file
        log_file.write(f"\nSummary:\n")
        if l1_mean is not None and l2_mean is not None:
            log_file.write(f"Chamfer-L1 - Mean: {l1_mean:.8f} ± {l1_std:.8f}\n")
            log_file.write(f"Chamfer-L2 - Mean: {l2_mean:.8f} ± {l2_std:.8f}\n")
    
    print(f"\nResults written to log.txt")
    print(f"Successfully processed {len(chamfer_l1_values)} files")

if __name__ == "__main__":
    main()