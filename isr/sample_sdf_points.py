import torch
import numpy as np
import open3d as o3d
import os
import argparse
from pathlib import Path
import utils
import models


def load_model(model_path, conf, device='cpu'):
    """Load trained SDF model."""
    # Create network (same as training script)
    network = models.NPullNetwork(**conf['model.sdf_network'])
    bias = 0.5
    network.lin8 = torch.nn.Linear(in_features=256, out_features=2, bias=True)
    
    # Load weights and set to eval mode
    network.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    network.to(device).eval()
    return network


@torch.no_grad()
def get_sdf_values(network, points, device='cpu'):
    """Get SDF values for points. Negative = inside, positive = outside."""
    points = points.to(device)
    out = network.sdf(points).softmax(1)
    return -(out[:, 1] - out[:, 0]).cpu()


def sample_batch(bounds, n_points):
    """Sample a batch of random points."""
    bound_min, bound_max = bounds
    points = torch.rand(n_points, 3, dtype=torch.float32)
    points = points * (torch.tensor(bound_max - bound_min, dtype=torch.float32)) + torch.tensor(bound_min, dtype=torch.float32)
    return points


def sample_neighborhood(seed_points, bounds, radius, n_points):
    """Sample points in neighborhoods around seed points."""
    bound_min, bound_max = bounds
    
    # Pick random seed points
    n_seeds = min(len(seed_points), n_points // 10)
    if n_seeds == 0:
        return torch.empty(0, 3, dtype=torch.float32)
    
    # Convert to numpy array if it's a list
    if isinstance(seed_points, list):
        seed_points = np.array(seed_points)
    
    seed_idx = np.random.choice(len(seed_points), n_seeds, replace=False)
    selected_seeds = seed_points[seed_idx]
    
    # Sample around each seed
    points_per_seed = n_points // n_seeds
    neighborhood_points = []
    
    for seed in selected_seeds:
        # Gaussian noise around seed point
        noise = torch.randn(points_per_seed, 3, dtype=torch.float32) * radius
        points = torch.tensor(seed, dtype=torch.float32) + noise
        
        # Clamp to bounds
        points = torch.clamp(points, 
                           torch.tensor(bound_min, dtype=torch.float32), 
                           torch.tensor(bound_max, dtype=torch.float32))
        neighborhood_points.append(points)
    
    return torch.cat(neighborhood_points) if neighborhood_points else torch.empty(0, 3, dtype=torch.float32)


def adaptive_sample(network, bounds, n_inside=10000, n_outside=10000, device='cpu'):
    """Adaptively sample exactly n_inside and n_outside points."""
    inside_points = []
    outside_points = []
    
    batch_size = 5000
    max_attempts = 500
    
    for attempt in range(max_attempts):
        if len(inside_points) >= n_inside and len(outside_points) >= n_outside:
            break
        
        # Phase 1: Random sampling (always do some)
        if attempt < 5:
            points = sample_batch(bounds, batch_size)
        else:
            # Phase 2: Neighborhood sampling
            # 50% random, 25% around inside seeds, 25% around outside seeds
            n_random = batch_size // 2
            n_inside_neigh = batch_size // 4
            n_outside_neigh = batch_size // 4
            
            random_points = sample_batch(bounds, n_random)
            
            # Sample around existing inside points
            inside_neigh = sample_neighborhood(inside_points, bounds, 
                                             radius=0.1, n_points=n_inside_neigh)
            
            # Sample around existing outside points  
            outside_neigh = sample_neighborhood(outside_points, bounds, 
                                              radius=0.1, n_points=n_outside_neigh)
            
            # Combine all points
            points = torch.cat([random_points, inside_neigh, outside_neigh])
        
        # Evaluate SDF
        sdf_values = get_sdf_values(network, points, device)
        
        # Collect points we still need
        if len(inside_points) < n_inside:
            new_inside = points[sdf_values < 0].numpy()
            needed = n_inside - len(inside_points)
            inside_points.extend(new_inside[:needed])
        
        if len(outside_points) < n_outside:
            new_outside = points[sdf_values >= 0].numpy()
            needed = n_outside - len(outside_points)
            outside_points.extend(new_outside[:needed])
        
        print(f"Attempt {attempt+1}: {len(inside_points)} inside, {len(outside_points)} outside")
    
    return np.array(inside_points[:n_inside]), np.array(outside_points[:n_outside])


def save_points(points, base_path):
    """Save points as both .npy and .ply files."""
    # Save as numpy
    np.save(f"{base_path}.npy", points)
    
    # Save as PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(f"{base_path}.ply", pcd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', required=True, help='Path to .pth model')
    parser.add_argument('--config', '-c', required=True, help='Config file used for training')
    parser.add_argument('--n_points', '-n', type=int, default=120000, help='Total points to generate')
    parser.add_argument('--bounds', nargs=6, type=float, default=[-1,1,-1,1,-1,1], 
                       help='Bounds: xmin xmax ymin ymax zmin zmax')
    parser.add_argument('--output_dir', '-o', default='inputs', help='Output directory')
    parser.add_argument('--device', '-d', default='auto', help='Device: auto, cpu, mps, cuda')
    
    args = parser.parse_args()
    
    # Setup
    np.random.seed(42)
    torch.manual_seed(42)
    conf = utils.load_conf(args.config)
    
    # Auto-detect device or use specified
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            # Enable CPU fallback for MPS
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
        if device == 'mps':
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model: {args.model_path}")
    network = load_model(args.model_path, conf, device)
    
    # Get bounds
    bound_min = np.array(args.bounds[::2])
    bound_max = np.array(args.bounds[1::2])
    print(f"Sampling in bounds: {bound_min} to {bound_max}")
    
    # Sample exactly 10k inside and 10k outside points
    print(f"Sampling exactly 10000 inside and 10000 outside points...")
    inside_points, outside_points = adaptive_sample(network, (bound_min, bound_max), 
                                                   n_inside=10000, n_outside=10000, device=device)
    
    print(f"Final: {len(inside_points)} inside, {len(outside_points)} outside points")
    
    # Save files
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = Path(args.model_path).stem.replace('model_', '')
    base_name = f"{model_name}_20k"  # Fixed to 20k total (10k inside + 10k outside)
    
    save_points(inside_points, f"{args.output_dir}/{base_name}_Xtrain_in")
    save_points(outside_points, f"{args.output_dir}/{base_name}_Xtrain_out")
    
    print(f"Saved: {base_name}_Xtrain_in/out.npy and .ply (10k points each)")


if __name__ == '__main__':
    main()