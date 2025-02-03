import pandas as pd
import re
import os

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
    
    # Read CSV file with explicit delimiter
    df = pd.read_csv(csv_file)    
    
    results = {}
    
    # Analyze each file
    for filename, endpoints in endpoint_data.items():
        # Extract file ID from filename
        ribfrac_id = os.path.splitext(os.path.basename(filename))[0]
        ribfrac_id = ribfrac_id.split('-')[0]
        
        # Find the corresponding rib scan
        file_row = df[df['public_id'] == ribfrac_id].iloc[0]
        
        #import pdb; pdb.set_trace()
        
        # Count total unique label codes (excluding 0)
        total_fractures = len(df[df['public_id'] == ribfrac_id][df['label_code'] != 0])
        
        # Count total non-zero endpoints
        total_endpoints = sum(1 for ep in endpoints if ep > 0)
        
        # Get total skeletons
        total_skeletons = skeleton_data[filename]
        
        results[filename] = {
            'total_fractures': total_fractures,
            'total_endpoints': total_endpoints,
            'total_skeletons': total_skeletons,
            'match': total_fractures == total_endpoints
        }
    
    return results


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
            print()
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check the file paths and ensure they are correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
    