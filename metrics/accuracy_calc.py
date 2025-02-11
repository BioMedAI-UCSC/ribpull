import pandas as pd
import matplotlib.pyplot as plt
import re

def extract_fracture_counts(file_path):
    """Extract the number of potential fractures for each RibFrac from the analysis summary."""
    fracture_counts = {}
    current_file = None
    
    with open(file_path, 'r') as file:
        for line in file:
            # Look for file name
            file_match = re.search(r'File: (RibFrac\d+)', line)
            if file_match:
                current_file = file_match.group(1)
            
            # Look for number of potential fractures
            frac_match = re.search(r'Number of potential fractures: (\d+)', line)
            if frac_match and current_file:
                fracture_counts[current_file] = int(frac_match.group(1))
    
    return fracture_counts

def count_ground_truth_fractures(csv_path):
    """Count the number of actual fractures (non-zero label_codes) for each RibFrac."""
    df = pd.read_csv(csv_path)
    
    # Group by public_id and count non-zero label_codes
    ground_truth = df[df['label_code'] != 0].groupby('public_id').size()
    
    # Convert index to match the format in the analysis summary
    ground_truth.index = ground_truth.index.str.split('-').str[0]
    
    return ground_truth

def create_ratio_plot(ground_truth, calculated):
    """Create and save a plot showing the ratio of ground truth to calculated fractures."""
    # Create a DataFrame with both counts
    comparison = pd.DataFrame({
        'ground_truth': ground_truth,
        'calculated': pd.Series(calculated)
    })
    
    # Keep only entries that have both ground truth and calculated values
    comparison = comparison.dropna()
    
    # Calculate ratios
    comparison['ratio'] = comparison['ground_truth'] / comparison['calculated']
    
    # Sort the index numerically
    comparison.index = pd.to_numeric(comparison.index.str.replace('RibFrac', ''))
    comparison = comparison.sort_index()
    comparison.index = 'RibFrac' + comparison.index.astype(str)
    
    # Create the plot with more width to accommodate labels
    plt.figure(figsize=(12, 6))
    
    # Create bars with more spacing
    bars = plt.bar(range(len(comparison)), comparison['ratio'])
    
    # Customize the plot
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect Match (ratio=1)')
    plt.title('Ratio of Ground Truth to Calculated Fractures per RibFrac')
    plt.xlabel('RibFrac ID')
    plt.ylabel('Ground Truth / Calculated Fractures')
    
    # Set x-ticks at bar positions with RibFrac labels
    plt.xticks(range(len(comparison)), comparison.index, rotation=45, ha='right')
    
    # Add gridlines
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig('fracture_ratio_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print the data used in the plot
    print("\nData used in plot:")
    print("------------------")
    print(comparison.to_string())

def main():
    # Extract calculated fracture counts from the analysis summary
    calculated_counts = extract_fracture_counts('analysis_summary.txt')
    
    # Get ground truth counts from the CSV
    ground_truth_counts = count_ground_truth_fractures('ribfrac_total_info.csv')
    
    # Create and save the plot
    create_ratio_plot(ground_truth_counts, calculated_counts)

if __name__ == "__main__":
    main()