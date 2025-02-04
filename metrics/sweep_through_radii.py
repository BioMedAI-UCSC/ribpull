import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
script_path = "../utils/endpoint_detector.py"  # Change this to the filename of your provided script
directory_path = "../test/ribcages/skeletons_gt"  # Change to your data directory
output_root = Path("radius_weight_results")
output_root.mkdir(exist_ok=True)

# Range of radius_weight values to test
radius_weights = np.arange(1.5, 3.6, 0.1)

# Storage for results
results = {}

for radius_weight in radius_weights:
    print(f"\nRunning analysis with radius_weight = {radius_weight:.2f}")
    
    # Define output directory for this run
    output_dir = output_root / f"radius_{radius_weight:.2f}"
    output_dir.mkdir(exist_ok=True)

    # Run the script with the specific radius_weight
    subprocess.run([
        "python", script_path, directory_path, "--radius", str(radius_weight)
    ], check=True)

    # Move analysis summary to separate folder
    analysis_summary_path = Path(directory_path) / "endpoint_analysis" / "analysis_summary.txt"
    if analysis_summary_path.exists():
        output_analysis_summary = output_dir / "analysis_summary.txt"
        analysis_summary_path.rename(output_analysis_summary)
    
    # Parse the analysis summary file
    if output_analysis_summary.exists():
        with open(output_analysis_summary, "r") as f:
            lines = f.readlines()
        
        num_skeletons = []
        num_endpoints = []
        for line in lines:
            if "Number of skeletons:" in line:
                num_skeletons.append(int(line.split(":")[1].strip()))
            elif "endpoints" in line and "Skeleton" in line:
                num_endpoints.append(int("".join(filter(str.isdigit, line.split(":")[1]))))
        
        avg_skeletons = np.mean(num_skeletons) if num_skeletons else 0
        avg_endpoints = np.mean(num_endpoints) if num_endpoints else 0
        
        results[radius_weight] = {
            "avg_skeletons": avg_skeletons,
            "avg_endpoints": avg_endpoints
        }

# Convert results into sorted lists
radius_values = sorted(results.keys())
avg_skeletons_list = [results[r]["avg_skeletons"] for r in radius_values]
avg_endpoints_list = [results[r]["avg_endpoints"] for r in radius_values]

# Plot the results
plt.figure(figsize=(10, 5))

# Plot 1: Average number of skeletons per ribcage
plt.figure(figsize=(8, 5))
plt.plot(radius_weights, avg_skeletons_list, marker="o", linestyle="-", color="b")
plt.xlabel("Radius Weight")
plt.ylabel("Avg. Skeletons per Ribcage")
plt.title("Effect of Radius Weight on Skeleton Count")
plt.grid(True)
plt.savefig("average_skeletons_per_ribcage.png")  # Save as PNG
plt.close()  # Close the figure to free memory

# Plot 2: Average number of endpoints per ribcage
plt.figure(figsize=(8, 5))
plt.plot(radius_weights, avg_endpoints_list, marker="s", linestyle="-", color="r")
plt.xlabel("Radius Weight")
plt.ylabel("Avg. Endpoints per Ribcage")
plt.title("Effect of Radius Weight on Endpoints Count")
plt.grid(True)
plt.savefig("average_endpoints_per_ribcage.png")  # Save as PNG
plt.close()  # Close the figure to free memory