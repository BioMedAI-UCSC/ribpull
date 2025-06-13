import numpy as np
import matplotlib.pyplot as plt

def compute_sdf_gradient(sdf_func, points, h=1e-5):
    """
    Compute gradient of SDF using finite differences
    
    Args:
        sdf_func: function that takes (x,y,z) and returns signed distance
        points: Nx3 array of query points
        h: step size for finite differences
    
    Returns:
        Nx3 array of gradient vectors
    """
    gradients = np.zeros_like(points)
    
    for i in range(3):  # x, y, z directions
        # Create offset vectors
        offset = np.zeros(3)
        offset[i] = h
        
        # Finite difference approximation
        points_plus = points + offset
        points_minus = points - offset
        
        sdf_plus = np.array([sdf_func(*p) for p in points_plus])
        sdf_minus = np.array([sdf_func(*p) for p in points_minus])
        
        gradients[:, i] = (sdf_plus - sdf_minus) / (2 * h)
    
    return gradients

def check_gradient_properties(gradients, tolerance=0.01):
    """
    Check if gradients have unit magnitude (SDF property)
    """
    # Compute magnitudes
    magnitudes = np.linalg.norm(gradients, axis=1)
    
    # Check how many are close to 1.0
    unit_mask = np.abs(magnitudes - 1.0) <= tolerance
    percentage_valid = np.mean(unit_mask) * 100
    
    print(f"Gradient magnitude statistics:")
    print(f"  Mean: {np.mean(magnitudes):.4f}")
    print(f"  Std:  {np.std(magnitudes):.4f}")
    print(f"  Min:  {np.min(magnitudes):.4f}")
    print(f"  Max:  {np.max(magnitudes):.4f}")
    print(f"  {percentage_valid:.1f}% within tolerance of 1.0")
    
    return magnitudes, unit_mask


def validate_sdf_gradients_quick(model, device, bounds, num_test_points=50):
    """Quick SDF gradient validation"""
    def sdf_func(x, y, z):
        with torch.no_grad():
            point = torch.tensor([[x, y, z]], dtype=torch.float32).to(device)
            out = model.sdf(point)
            return -(out.softmax(1)[0, 1] - out.softmax(1)[0, 0]).cpu().numpy().item()
    
    # Random test points
    test_points = np.random.uniform(bounds[0], bounds[1], (num_test_points, 3))
    gradients = compute_sdf_gradient(sdf_func, test_points)
    magnitudes, valid_mask = check_gradient_properties(gradients)
    
    print(f"SDF Gradient Check: {np.mean(valid_mask)*100:.1f}% valid (mean mag: {np.mean(magnitudes):.3f})")
    return np.mean(valid_mask) > 0.8