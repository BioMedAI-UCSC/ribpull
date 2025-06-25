import torch
import numpy as np
import matplotlib.pyplot as plt
from models import NPullNetwork
import utils

def load_trained_model(model_path, config_path):
    device = 'cpu'
    
    # Load config
    conf = utils.load_conf(config_path)
    
    # Initialize network
    sdf_network = NPullNetwork(**conf['model.sdf_network'])
    
    # Apply spherical_init EXACTLY like in training
    bias = 0.5
    sdf_network.lin8 = torch.nn.Linear(in_features=256, out_features=2, bias=True)
    torch.nn.init.normal_(sdf_network.lin8.weight[0], mean=np.sqrt(np.pi) / np.sqrt(256), std=0.0001)
    torch.nn.init.constant_(sdf_network.lin8.bias[0], -bias)
    torch.nn.init.normal_(sdf_network.lin8.weight[1], mean=-np.sqrt(np.pi) / np.sqrt(256), std=0.0001)
    torch.nn.init.constant_(sdf_network.lin8.bias[1], bias)
    
    # Load the saved weights
    state_dict = torch.load(model_path, map_location=device)
    sdf_network.load_state_dict(state_dict)
    sdf_network.to(device)
    sdf_network.eval()
    
    return sdf_network, device

def query_sdf_single(model, device, point):
    """Query SDF value for a single point"""
    with torch.no_grad():
        point_tensor = torch.tensor([point], dtype=torch.float32).to(device)
        out = model.sdf(point_tensor)
        sdf_value = -(out.softmax(1)[0, 1] - out.softmax(1)[0, 0])
        return sdf_value.cpu().numpy().item()

def compute_sdf_gradient(model, device, points, h=1e-2):
    """Compute gradient of SDF using finite differences"""
    gradients = np.zeros_like(points)
    
    for i in range(3):  # x, y, z directions
        # Create offset vectors
        offset = np.zeros(3)
        offset[i] = h
        
        # Finite difference approximation
        points_plus = points + offset
        points_minus = points - offset
        
        # Query SDF for each point
        sdf_plus = np.array([query_sdf_single(model, device, p) for p in points_plus])
        sdf_minus = np.array([query_sdf_single(model, device, p) for p in points_minus])
        
        gradients[:, i] = (sdf_plus - sdf_minus) / (2 * h)
    
    return gradients

def classify_points_by_region(sdf_values, surface_threshold=0.1):
    """Classify points into regions based on SDF values"""
    regions = []
    for sdf in sdf_values:
        if abs(sdf) <= surface_threshold:
            regions.append('Surface')
        elif sdf > surface_threshold:
            regions.append('Outside')
        else:
            regions.append('Inside')
    return np.array(regions)

def generate_test_points_by_region(model, device, bounds_min, bounds_max, 
                                 points_per_region=50, surface_threshold=0.1, max_attempts=2000):
    """Generate points in different regions: inside, surface, outside"""
    points = {'Surface': [], 'Inside': [], 'Outside': []}
    sdf_values = {'Surface': [], 'Inside': [], 'Outside': []}
    
    print("Searching for points in different regions...")
    
    for attempt in range(max_attempts):
        # Sample random point
        point = np.random.uniform(bounds_min, bounds_max, 3)
        sdf_val = query_sdf_single(model, device, point)
        
        # Classify point
        if abs(sdf_val) <= surface_threshold and len(points['Surface']) < points_per_region:
            region = 'Surface'
        elif sdf_val > surface_threshold and len(points['Outside']) < points_per_region:
            region = 'Outside'
        elif sdf_val < -surface_threshold and len(points['Inside']) < points_per_region:
            region = 'Inside'
        else:
            continue
            
        points[region].append(point)
        sdf_values[region].append(sdf_val)
        
        if attempt % 200 == 0:
            print(f"  Attempt {attempt}: Surface={len(points['Surface'])}, "
                  f"Inside={len(points['Inside'])}, Outside={len(points['Outside'])}")
        
        # Check if we have enough points in all regions
        if (len(points['Surface']) >= points_per_region and 
            len(points['Inside']) >= points_per_region and 
            len(points['Outside']) >= points_per_region):
            break
    
    # Convert to numpy arrays
    for region in points:
        if len(points[region]) > 0:
            points[region] = np.array(points[region])
            sdf_values[region] = np.array(sdf_values[region])
        else:
            points[region] = np.empty((0, 3))
            sdf_values[region] = np.array([])
    
    print(f"\nFound points: Surface={len(points['Surface'])}, "
          f"Inside={len(points['Inside'])}, Outside={len(points['Outside'])}")
    
    return points, sdf_values

def analyze_gradient_quality(gradients, region_name):
    """Analyze gradient quality for a specific region"""
    if len(gradients) == 0:
        return None
        
    magnitudes = np.linalg.norm(gradients, axis=1)
    
    # Statistics
    stats = {
        'mean_magnitude': np.mean(magnitudes),
        'std_magnitude': np.std(magnitudes),
        'min_magnitude': np.min(magnitudes),
        'max_magnitude': np.max(magnitudes),
        'median_magnitude': np.median(magnitudes),
        'near_unit_percentage': np.mean(np.abs(magnitudes - 1.0) <= 0.1) * 100
    }
    
    return stats, magnitudes

def create_gradient_report(model_path, config_path, output_file='sdf_gradient_report.png', 
                         save_plot=True, show_plot=False):
    """Generate comprehensive gradient analysis report"""
    print("=== SDF Gradient Quality Analysis Report ===\n")
    
    # Load model
    print("Loading model...")
    model, device = load_trained_model(model_path, config_path)
    
    # Define bounds (from your MeshLab analysis)
    bounds_min = np.array([0.1, 0.2, 0.1])
    bounds_max = np.array([1.1, 0.9, 1.2])
    
    # Define regions and colors
    regions = ['Surface', 'Inside', 'Outside']
    colors = ['red', 'blue', 'green']
    
    # Generate points in different regions
    points_dict, sdf_dict = generate_test_points_by_region(
        model, device, bounds_min, bounds_max, points_per_region=30
    )
    
    # Check which regions we actually found
    available_regions = [region for region in regions if len(points_dict[region]) > 0]
    
    if not available_regions:
        print("ERROR: No points found in any region!")
        return None
    
    print(f"Found data for regions: {available_regions}")
    
    # Only create plots if requested
    if save_plot or show_plot:
        # Create subplot grid based on available regions
        n_regions = len(available_regions)
        fig, axes = plt.subplots(2, n_regions, figsize=(5*n_regions, 10))
        
        # Handle case where we only have one region (axes won't be 2D)
        if n_regions == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('SDF Gradient Analysis by Region', fontsize=16, fontweight='bold')
    
    # Analyze each available region
    results = {}
    plot_idx = 0
    
    for region, color in zip(regions, colors):
        if len(points_dict[region]) == 0:
            print(f"\n{region} Region: No points found!")
            continue
            
        print(f"\n=== {region.upper()} REGION ANALYSIS ===")
        print(f"Number of points: {len(points_dict[region])}")
        print(f"SDF range: [{np.min(sdf_dict[region]):.4f}, {np.max(sdf_dict[region]):.4f}]")
        
        # Compute gradients
        gradients = compute_sdf_gradient(model, device, points_dict[region])
        
        # Analyze quality
        stats, magnitudes = analyze_gradient_quality(gradients, region)
        results[region] = {'stats': stats, 'magnitudes': magnitudes, 'sdf_values': sdf_dict[region]}
        
        # Print statistics
        print(f"Gradient Magnitude Statistics:")
        print(f"  Mean: {stats['mean_magnitude']:.4f}")
        print(f"  Std:  {stats['std_magnitude']:.4f}")
        print(f"  Min:  {stats['min_magnitude']:.4f}")
        print(f"  Max:  {stats['max_magnitude']:.4f}")
        print(f"  Median: {stats['median_magnitude']:.4f}")
        print(f"  Within ±0.1 of 1.0: {stats['near_unit_percentage']:.1f}%")
        
        # Create plots only if requested
        if save_plot or show_plot:
            # Plot histogram of gradient magnitudes
            axes[0, plot_idx].hist(magnitudes, bins=20, color=color, alpha=0.7, edgecolor='black')
            axes[0, plot_idx].axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Ideal (|∇| = 1)')
            axes[0, plot_idx].set_title(f'{region} Region\n({len(magnitudes)} points)')
            axes[0, plot_idx].set_xlabel('Gradient Magnitude')
            axes[0, plot_idx].set_ylabel('Count')
            axes[0, plot_idx].legend()
            axes[0, plot_idx].grid(True, alpha=0.3)
            
            # Plot SDF vs Gradient Magnitude scatter
            axes[1, plot_idx].scatter(sdf_dict[region], magnitudes, color=color, alpha=0.6, s=30)
            axes[1, plot_idx].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Ideal (|∇| = 1)')
            axes[1, plot_idx].set_xlabel('SDF Value')
            axes[1, plot_idx].set_ylabel('Gradient Magnitude')
            axes[1, plot_idx].set_title(f'SDF vs Gradient Magnitude')
            axes[1, plot_idx].legend()
            axes[1, plot_idx].grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Save and/or show plots
    if save_plot or show_plot:
        plt.tight_layout()
        if save_plot:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"\nReport saved as: {output_file}")
        if show_plot:
            plt.show()
        else:
            plt.close()  # Close figure to free memory if not showing
    
    # Summary comparison
    print(f"\n=== SUMMARY COMPARISON ===")
    print(f"{'Region':<10} {'Points':<8} {'Mean |∇|':<10} {'Std |∇|':<10} {'Near Unit %':<12}")
    print("-" * 55)
    
    for region in regions:
        if region in results and results[region]['stats'] is not None:
            stats = results[region]['stats']
            print(f"{region:<10} {len(results[region]['magnitudes']):<8} "
                  f"{stats['mean_magnitude']:<10.3f} {stats['std_magnitude']:<10.3f} "
                  f"{stats['near_unit_percentage']:<12.1f}%")
    
    return results

if __name__ == "__main__":
    # Generate the report
    model_path = "meshes/model_20000.pth"
    config_path = "configs/ribcage.conf"
    
    results = create_gradient_report(model_path, config_path, 
                                   save_plot=True, show_plot=False)