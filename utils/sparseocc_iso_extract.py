import torch
import numpy as np
from skimage import measure
import trimesh
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

def query_sdf_batch(model, device, points, batch_size=10000):
    """Query SDF values for a batch of points"""
    sdf_values = []
    
    with torch.no_grad():
        for i in range(0, len(points), batch_size):
            batch_points = torch.tensor(points[i:i+batch_size], dtype=torch.float32).to(device)
            out = model.sdf(batch_points)
            sdf_batch = -(out.softmax(1)[:, 1] - out.softmax(1)[:, 0])
            sdf_values.append(sdf_batch.cpu().numpy())
    
    return np.concatenate(sdf_values)

def high_resolution_adaptive_sampling(model, device, bounds, base_resolution=64, max_depth=3, surface_threshold=0.02):
    """
    Memory-efficient adaptive sampling with controlled resolution
    """
    print(f"=== MEMORY-EFFICIENT ADAPTIVE SAMPLING ===")
    print(f"Base resolution: {base_resolution}")
    print(f"Max depth: {max_depth}")
    print(f"Surface threshold: {surface_threshold}")
    
    # Calculate theoretical max to prevent memory issues
    theoretical_max = base_resolution * (2 ** max_depth)
    estimated_points = theoretical_max ** 3
    estimated_memory_gb = estimated_points * 8 / (1024**3)  # 8 bytes per float64
    
    print(f"Theoretical max resolution: {theoretical_max}")
    print(f"Estimated memory usage: {estimated_memory_gb:.1f} GB")
    
    if estimated_memory_gb > 4:  # Safety check
        print("WARNING: High memory usage expected!")
        # Automatically reduce if too high
        while estimated_memory_gb > 4 and max_depth > 1:
            max_depth -= 1
            theoretical_max = base_resolution * (2 ** max_depth)
            estimated_points = theoretical_max ** 3
            estimated_memory_gb = estimated_points * 8 / (1024**3)
        print(f"Auto-reduced to max_depth={max_depth}, estimated memory: {estimated_memory_gb:.1f} GB")
    
    # Phase 1: Find surface regions with moderate resolution
    print("\nPhase 1: Finding surface regions...")
    coarse_res = base_resolution // 8  # Start quite coarse for region detection
    x_coarse = np.linspace(bounds[0], bounds[1], coarse_res)
    y_coarse = np.linspace(bounds[0], bounds[1], coarse_res)
    z_coarse = np.linspace(bounds[0], bounds[1], coarse_res)
    
    interesting_regions = []
    
    for i in range(coarse_res - 1):
        for j in range(coarse_res - 1):
            for k in range(coarse_res - 1):
                # Sample each coarse cube more thoroughly
                test_res = 4  # 4x4x4 = 64 test points per coarse cube
                x_test = np.linspace(x_coarse[i], x_coarse[i+1], test_res)
                y_test = np.linspace(y_coarse[j], y_coarse[j+1], test_res)
                z_test = np.linspace(z_coarse[k], z_coarse[k+1], test_res)
                
                X_test, Y_test, Z_test = np.meshgrid(x_test, y_test, z_test, indexing='ij')
                test_points = np.stack([X_test.ravel(), Y_test.ravel(), Z_test.ravel()], axis=1)
                
                sdf_vals = query_sdf_batch(model, device, test_points)
                
                # More sensitive surface detection
                if sdf_vals.min() <= surface_threshold and sdf_vals.max() >= -surface_threshold:
                    interesting_regions.append({
                        'bounds': ((x_coarse[i], x_coarse[i+1]), 
                                  (y_coarse[j], y_coarse[j+1]), 
                                  (z_coarse[k], z_coarse[k+1])),
                        'min_sdf': sdf_vals.min(),
                        'max_sdf': sdf_vals.max()
                    })
    
    print(f"Found {len(interesting_regions)} surface regions out of {(coarse_res-1)**3} total regions")
    
    # Phase 2: High-resolution sampling of interesting regions
    print("\nPhase 2: High-resolution sampling of surface regions...")
    
    # Calculate target resolution for final grid (with memory safety)
    final_resolution = base_resolution * (2 ** max_depth)
    max_safe_resolution = 1024  
    final_resolution = min(final_resolution, max_safe_resolution)
    print(f"Target final grid resolution: {final_resolution}³ (capped for safety)")
    
    # Create full coordinate arrays for final grid
    x_full = np.linspace(bounds[0], bounds[1], final_resolution)
    y_full = np.linspace(bounds[0], bounds[1], final_resolution)
    z_full = np.linspace(bounds[0], bounds[1], final_resolution)
    
    # Initialize SDF grid with high values (outside surface)
    sdf_grid = np.ones((final_resolution, final_resolution, final_resolution)) * 2.0
    
    # Sample each interesting region with maximum density
    for region_idx, region in enumerate(interesting_regions):
        print(f"  Processing region {region_idx+1}/{len(interesting_regions)} - SDF range: [{region['min_sdf']:.3f}, {region['max_sdf']:.3f}]")
        
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = region['bounds']
        
        # Find corresponding indices in full grid
        i_start = np.searchsorted(x_full, x_min)
        i_end = np.searchsorted(x_full, x_max)
        j_start = np.searchsorted(y_full, y_min)
        j_end = np.searchsorted(y_full, y_max)
        k_start = np.searchsorted(z_full, z_min)
        k_end = np.searchsorted(z_full, z_max)
        
        # Ensure we have at least some resolution
        i_end = max(i_end, i_start + 8)
        j_end = max(j_end, j_start + 8)
        k_end = max(k_end, k_start + 8)
        
        # Clamp to grid bounds
        i_end = min(i_end, final_resolution)
        j_end = min(j_end, final_resolution)
        k_end = min(k_end, final_resolution)
        
        if i_start < i_end and j_start < j_end and k_start < k_end:
            # Extract coordinates for this region
            x_region = x_full[i_start:i_end]
            y_region = y_full[j_start:j_end]
            z_region = z_full[k_start:k_end]
            
            # Create dense meshgrid for this region
            X_region, Y_region, Z_region = np.meshgrid(x_region, y_region, z_region, indexing='ij')
            region_points = np.stack([X_region.ravel(), Y_region.ravel(), Z_region.ravel()], axis=1)
            
            # Query SDF values for this region
            region_sdf = query_sdf_batch(model, device, region_points)
            region_sdf_grid = region_sdf.reshape(X_region.shape)
            
            # Insert into full grid
            sdf_grid[i_start:i_end, j_start:j_end, k_start:k_end] = region_sdf_grid
            
            print(f"    Sampled {len(region_points)} points, SDF range: [{region_sdf.min():.3f}, {region_sdf.max():.3f}]")
    
    print(f"\nPhase 2 complete. Final grid SDF range: [{sdf_grid.min():.3f}, {sdf_grid.max():.3f}]")
    
    return sdf_grid, (x_full, y_full, z_full)

def extract_mesh_adaptive(sdf_grid, coordinates, surface_percentile=10):
    """
    Extract mesh using adaptive level selection based on SDF distribution
    """
    # Analyze SDF distribution, but EXCLUDE empty space values
    min_val, max_val = sdf_grid.min(), sdf_grid.max()
    
    # Filter out the "empty space" values (we set these to 2.0)
    actual_sdf_values = sdf_grid[sdf_grid < 1.5]  # Only keep real SDF values
    
    if len(actual_sdf_values) == 0:
        print("ERROR: No actual SDF values found in grid!")
        return np.array([]), np.array([]), np.array([])
    
    # Calculate percentile from ACTUAL SDF values only
    surface_level = np.percentile(actual_sdf_values, surface_percentile)
    
    print(f"SDF Grid Analysis:")
    print(f"  Full grid range: [{min_val:.4f}, {max_val:.4f}]")
    print(f"  Actual SDF values: {len(actual_sdf_values):,} out of {sdf_grid.size:,} total")
    print(f"  Actual SDF range: [{actual_sdf_values.min():.4f}, {actual_sdf_values.max():.4f}]")
    print(f"  Using {surface_percentile}th percentile of ACTUAL values as surface level: {surface_level:.4f}")

    vertices, faces, normals, _ = measure.marching_cubes(
        sdf_grid, 
        level=surface_level, 
        spacing=(
            coordinates[0][1] - coordinates[0][0],
            coordinates[1][1] - coordinates[1][0], 
            coordinates[2][1] - coordinates[2][0]
        )
    )
    
    # Offset vertices to correct position
    vertices[:, 0] += coordinates[0][0]
    vertices[:, 1] += coordinates[1][0]
    vertices[:, 2] += coordinates[2][0]
    
    print(f"Marching cubes successful: {len(vertices)} vertices, {len(faces)} faces")
    return vertices, faces, normals

def generate_mesh(model_path, config_path, output_path, bounds=(-1.5, 1.5), 
                 base_resolution=64, max_depth=3, surface_threshold=0.02, surface_percentile=10):
    """
    Generate high-resolution mesh using adaptive sampling
    """
    print("="*80)
    print("HIGH-RESOLUTION ADAPTIVE MESH GENERATION")
    print("="*80)
    
    # Load model
    print("Loading trained model...")
    model, device = load_trained_model(model_path, config_path)
    
    # Adaptive sampling
    print("Starting adaptive sampling...")
    sdf_grid, coordinates = high_resolution_adaptive_sampling(
        model, device, bounds, base_resolution, max_depth, surface_threshold
    )
    
    # Extract mesh
    print("\nExtracting mesh with marching cubes...")
    vertices, faces, normals = extract_mesh_adaptive(sdf_grid, coordinates, surface_percentile)
    
    if len(vertices) > 0:
        # Create and save mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        mesh.export(output_path)
        
        print(f"\n{'='*80}")
        print(f"SUCCESS! Generated mesh saved to: {output_path}")
        print(f"Final mesh: {len(vertices):,} vertices, {len(faces):,} faces")
        print(f"{'='*80}")
        
        return mesh
    else:
        print("ERROR: No mesh could be generated")
        return None

if __name__ == "__main__":
    # Configuration
    model_path = "meshes/model_20000.pth"
    config_path = "configs/ribcage.conf"
    output_path = "high_res_ribcage.obj"
    
    # Generate high-resolution mesh
    mesh = generate_mesh(
        model_path=model_path,
        config_path=config_path,
        output_path=output_path,
        bounds=(-1.5, 1.5),
        base_resolution=64,        
        max_depth=4,               
        surface_threshold=0.02,    
        surface_percentile=1.5     
    )
    
    print(f"\nMesh statistics:")
    print(f"  Bounding box: {mesh.bounds}")
    print(f"  Volume: {mesh.volume:.4f}")
    print(f"  Surface area: {mesh.area:.4f}")
    print(f"  Watertight: {mesh.is_watertight}")