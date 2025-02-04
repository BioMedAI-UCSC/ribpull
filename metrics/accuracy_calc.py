import pandas as pd
import re
import os
import matplotlib.pyplot as plt

def parse_endpoint_file(file_path):
    """Parse the endpoint text file and extract endpoint counts for each skeleton."""
    endpoint_data = {}
    skeleton_data = {}
    current_file = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Check if line indicates a new file
            file_match = re.match(r'File: (.*)', line)
            if file_match:
                current_file = file_match.group(1)
                endpoint_data[current_file] = []
                skeleton_data[current_file] = []
                continue
            
            # Check for skeleton endpoint counts
            endpoint_match = re.match(r'Skeleton \d+: (\d+) endpoints', line)
            if endpoint_match and current_file:
                endpoint_data[current_file].append(int(endpoint_match.group(1)))

            # Get total number of skeletons
            skeleton_count_match = re.match(r'Number of skeletons: (\d+)', line)
            if skeleton_count_match and current_file:
                skeleton_data[current_file] = int(skeleton_count_match.group(1))
    
    return endpoint_data, skeleton_data

def analyze_fracture_endpoints(endpoint_file, csv_file):
    """Compare endpoint counts with fracture labels from CSV."""
    # Parse endpoint data
    endpoint_data, skeleton_data = parse_endpoint_file(endpoint_file)
    
    # Read CSV file
    df = pd.read_csv(csv_file)    
    
    results = {}
    
    # Analyze each file
    for filename, endpoints in endpoint_data.items():
        # Extract file ID from filename
        ribfrac_id = os.path.splitext(os.path.basename(filename))[0]
        ribfrac_id = ribfrac_id.split('-')[0]
        
        # Find the corresponding rib scan
        file_row = df[df['public_id'] == ribfrac_id].iloc[0]
        
        # Count total unique label codes (excluding 0)
        total_fractures = len(df[df['public_id'] == ribfrac_id][df['label_code'] != 0])
        
        # Count total non-zero endpoints
        total_endpoints = sum(1 for ep in endpoints if ep > 0)
        
        # Get total skeletons
        total_skeletons = skeleton_data[filename]
        
        # Compute ratio, avoiding division by zero
        ratio = (total_fractures / (total_endpoints - 24)) * 100 if (total_endpoints - 24) != 0 else 0
        
        results[filename] = {
            'total_fractures': total_fractures,
            'total_endpoints': total_endpoints,
            'total_skeletons': total_skeletons,
            'match': total_fractures == total_endpoints,
            'ratio': ratio
        }
    
    return results

def plot_results(results):
    """Generates and saves a bar plot of the fracture-to-endpoint ratio."""
    # Extract and sort data
    ribcage_data = sorted(results.items(), key=lambda x: int(re.search(r'\d+', x[0]).group()))
    
    ribcages = [re.search(r'RibFrac\d+', name).group() for name, _ in ribcage_data]
    ratios = [data['ratio'] for _, data in ribcage_data]

    # Create bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(ribcages, ratios, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Ratio of GT Fractures vs. Detected Endpoints")
    plt.xlabel("Ribcage Files")
    plt.title("Fracture Endpoint Analysis (Sorted)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plt.savefig("results.png")
    plt.show()
    print("Plot saved as 'results.png'.")

def main():
    # Paths to your files - modify these as needed
    endpoint_file = 'analysis_summary.txt'  # Replace with your endpoint file path
    csv_file = 'ribfrac_total_info.csv'  # Replace with your CSV file path
    
    try:
        # Perform analysis
        comparison_results = analyze_fracture_endpoints(endpoint_file, csv_file)
        
        # Print results
        print("Fracture Endpoint Analysis:")
        print("-" * 40)
        for filename, result in comparison_results.items():
            print(f"File: {filename}")
            print(f"  Total Fractures: {result['total_fractures']}")
            print(f"  Total Endpoints: {result['total_endpoints']}")
            print(f"  Total Skeletons: {result['total_skeletons']}")
            print(f"  Match: {'Yes' if result['match'] else 'No'}")
            print(f"  Ratio of GT Fractures vs. Detected Endpoints: {result['ratio']:.2f}")
            print()
        
        # Generate and save the plot
        plot_results(comparison_results)
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check the file paths and ensure they are correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
