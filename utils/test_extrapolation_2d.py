import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_closing, binary_erosion, label
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_sdf import create_extrapolated_volume, remove_small_components

def create_2d_toy_dataset(grid_size=100, num_points=100, shape_type='ribs', noise_level=0.1, random_seed=42):
    """
    Args:
        grid_size: Size of the 2D grid (grid_size x grid_size)
        num_points: Number of points to generate
        shape_type: 'circle' or 'rectangle'
        noise_level: Level of noise to add (0.0 to 1.0)
        random_seed: Random seed for reproducibility

    Returns:
        points: Nx2 array of coordinates
        labels: N-length array with 1 for foreground, 0 for background
        ground_truth: Binary grid representing the true shape
    """
    #np.random.seed(random_seed)
    
    # Create empty grid for ground truth
    ground_truth = np.zeros((grid_size, grid_size), dtype=bool)
    
    # Create shape
    center = grid_size // 2
    radius = grid_size // 4    
    if shape_type == 'circle':
        y, x = np.ogrid[:grid_size, :grid_size]
        dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
        ground_truth = dist_from_center <= radius
    elif shape_type == 'rectangle':
        start_x, end_x = center - radius, center + radius
        start_y, end_y = center - radius, center + radius
        ground_truth[start_y:end_y, start_x:end_x] = True
    elif shape_type == 'ribs':
        # Create two ellipsoid shapes like ribs, rotated along z-axis
        y, x = np.ogrid[:grid_size, :grid_size]
        
        # Parameters for the ellipsoids
        a, b = radius * 0.7, radius * 0.25  # Semi-major and semi-minor axes
        offset = radius * 1.2  # Offset from center
        rotation_angle = np.pi/6  # 30 degrees rotation
        
        # Function to rotate points
        def rotate_point(x, y, cx, cy, angle):
            # Translate point to origin
            x_t = x - cx
            y_t = y - cy
            
            # Rotate point
            x_r = x_t * np.cos(angle) - y_t * np.sin(angle)
            y_r = x_t * np.sin(angle) + y_t * np.cos(angle)
            
            # Translate back
            return x_r + cx, y_r + cy
        
        # Create meshgrid for easier calculations
        xx, yy = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        
        # Left rib (rotated clockwise)
        left_x = center - offset/2
        left_y = center
        
        # Rotate points
        left_xx, left_yy = rotate_point(xx, yy, left_x, left_y, rotation_angle)
        
        # Create left ellipsoid equation
        left_ellipse = (((left_xx - left_x)**2 / a**2) + 
                        ((left_yy - left_y)**2 / b**2)) <= 1.0
        
        # Right rib (rotated counter-clockwise)
        right_x = center + offset/2
        right_y = center
        
        # Rotate points
        right_xx, right_yy = rotate_point(xx, yy, right_x, right_y, -rotation_angle)
        
        # Create right ellipsoid equation
        right_ellipse = (((right_xx - right_x)**2 / a**2) + 
                         ((right_yy - right_y)**2 / b**2)) <= 1.0
        
        # Combine the two ellipsoids (ensure no overlap)
        ground_truth = left_ellipse | right_ellipse
        
        # Check for overlap and adjust if needed
        if np.any(left_ellipse & right_ellipse):
            print("Warning: Ellipsoids overlap - adjusting offset")
            ground_truth = np.zeros_like(ground_truth)
            
            # Try increasing offset until no overlap
            while np.any(left_ellipse & right_ellipse):
                offset *= 1.1
                
                # Recalculate positions
                left_x = center - offset/2
                right_x = center + offset/2
                
                # Rotate points
                left_xx, left_yy = rotate_point(xx, yy, left_x, left_y, rotation_angle)
                right_xx, right_yy = rotate_point(xx, yy, right_x, right_y, -rotation_angle)
                
                # Recalculate ellipses
                left_ellipse = (((left_xx - left_x)**2 / a**2) + 
                                ((left_yy - left_y)**2 / b**2)) <= 1.0
                right_ellipse = (((right_xx - right_x)**2 / a**2) + 
                                 ((right_yy - right_y)**2 / b**2)) <= 1.0
            
            ground_truth = left_ellipse | right_ellipse
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")

    # Sample from the foreground with some background points
    foreground_ratio = 0.9  
    num_foreground = int(num_points * foreground_ratio)
    num_background = num_points - num_foreground
    
    # Generate foreground & background points and combine
    fg_indices = np.where(ground_truth)
    fg_indices = np.array(fg_indices).T
    fg_indices_sample = fg_indices[np.random.choice(len(fg_indices), num_foreground, replace=True)]
    bg_indices = np.where(~ground_truth)
    bg_indices = np.array(bg_indices).T
    bg_indices_sample = bg_indices[np.random.choice(len(bg_indices), num_background, replace=True)]
    points = np.vstack([fg_indices_sample, bg_indices_sample]) # combines background and foreground
    
    # Create labels (1 for foreground, 0 for background)
    labels = np.zeros(num_points, dtype=int)
    labels[:num_foreground] = 1
    
    # Shuffle points and labels and add noise
    shuffle_idx = np.random.permutation(num_points)
    points = points[shuffle_idx]
    labels = labels[shuffle_idx]
    noise = np.random.normal(0, noise_level * radius, points.shape)
    points = points + noise
    
    # Clamp points to grid boundaries
    points = np.clip(points, 0, grid_size - 1)
    
    return points, labels, ground_truth

def create_extrapolated_volume_2d(points, labels, volume_shape, radius):
    """
    Args:
        points: Nx2 array of coordinates
        labels: N-length array with 1 for foreground, 0 for background
        volume_shape: Tuple of (height, width) for 2D grid size
        radius: Influence radius for each point
    
    Returns:
        Probability grid (float array between 0 and 1)
    """
    # Create empty volume
    volume = np.zeros(volume_shape, dtype=float)
    
    # Convert to integer coordinates for grid
    point_coords = np.round(points).astype(int)
    
    # Ensure all points are within bounds
    valid_mask = (
        (point_coords[:, 0] >= 0) & (point_coords[:, 0] < volume_shape[0]) &
        (point_coords[:, 1] >= 0) & (point_coords[:, 1] < volume_shape[1])
    )
    
    point_coords = point_coords[valid_mask]
    point_labels = labels[valid_mask]
    
    # Focus only on foreground points (more efficient)
    foreground_points = point_coords[point_labels == 1]
    
    if len(foreground_points) == 0:
        print("Warning: No valid foreground points in volume")
        return volume
    
    # Process points in batches to improve efficiency
    radius_int = int(radius)
    radius_squared = radius**2
    batch_size = min(1000, len(foreground_points))  # Process 1000 points at a time
    
    for batch_start in range(0, len(foreground_points), batch_size):
        batch_end = min(batch_start + batch_size, len(foreground_points))
        batch_points = foreground_points[batch_start:batch_end]
        
        for point in batch_points:
            # Get bounding box around point
            y, x = point
            y_min, y_max = max(0, y-radius_int), min(volume_shape[0], y+radius_int+1)
            x_min, x_max = max(0, x-radius_int), min(volume_shape[1], x+radius_int+1)
            
            # Create grid for this subvolume
            y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]
            
            # Calculate squared distances
            squared_distances = (y_grid-y)**2 + (x_grid-x)**2
            
            # Apply influence based on distance
            influence = np.maximum(0, 1.0 - squared_distances / radius_squared)
            
            # Update the volume - use maximum influence at each pixel
            volume[y_min:y_max, x_min:x_max] = np.maximum(
                volume[y_min:y_max, x_min:x_max], influence
            )
    
    return volume

def remove_small_components_2d(binary_image, min_size=10):
    """
    2D version of remove_small_components function.
    """
    # Label connected components
    labeled_image, num_features = label(binary_image)
    
    if num_features == 0:
        return binary_image
        
    print(f"Found {num_features} connected components")
    
    # Count pixels in each component
    component_sizes = np.bincount(labeled_image.ravel())[1:] if num_features > 0 else []
    
    # Find small components
    small_components = np.where(component_sizes < min_size)[0] + 1  # +1 because background is 0
    
    # Count total pixels in small components
    small_pixels = sum(component_sizes[i-1] for i in small_components) if len(small_components) > 0 else 0
    
    if len(small_components) > 0:
        print(f"Removing {len(small_components)} small components ({small_pixels} pixels)")
        
        # Create mask of components to remove
        remove_mask = np.isin(labeled_image, small_components)
        
        # Remove small components
        cleaned_image = binary_image.copy()
        cleaned_image[remove_mask] = False
        
        return cleaned_image
    else:
        print("No small components to remove")
        return binary_image

def process_2d_point_cloud(points, labels, grid_shape, influence_radius, threshold, 
                         smooth_sigma, min_component_size, closing_kernel_size=3):
    """
    Process labeled 2D points to SDF using extrapolation.
    
    Args:
        points: Nx2 array of coordinates
        labels: N-length array with 1 for foreground, 0 for background
        grid_shape: Shape of the grid (height, width)
        influence_radius: Radius of influence for each point
        closing_kernel_size: Size of kernel for morphological closing
        threshold: Threshold for binarizing the probability grid
        smooth_sigma: Sigma for Gaussian smoothing (0 for no smoothing)
        min_component_size: Minimum size for connected components
    
    Returns:
        SDF as numpy array, binary grid, and probability grid
    """
    print("Creating extrapolated probability grid...")
    prob_grid = create_extrapolated_volume_2d(
        points, labels, grid_shape, influence_radius
    )
    print("Applying morphological closing...")
    binary_grid = prob_grid > threshold
    if closing_kernel_size > 0:
        kernel = np.ones((closing_kernel_size, closing_kernel_size))
        binary_grid = binary_closing(binary_grid, structure=kernel)
    print("Removing small disconnected components...")
    binary_grid = remove_small_components_2d(binary_grid, min_size=min_component_size)
    print("Computing SDF...")    
    print("Ensuring hollow structure...")
    # Erode by 2 pixels to remove interior
    core = binary_erosion(binary_grid, iterations=2)
    hollow_grid = binary_grid & ~core
    
    # Use hollow grid for SDF calculation
    print("Computing SDF from hollow structure...")
    outside_distance = distance_transform_edt(~hollow_grid)
    inside_distance = distance_transform_edt(hollow_grid)
    sdf = outside_distance - inside_distance
    
    print(f"Applying Gaussian smoothing with sigma={smooth_sigma}")
    sdf = gaussian_filter(sdf, sigma=smooth_sigma)
    
    return sdf, binary_grid, prob_grid

def visualize_results(points, labels, ground_truth, prob_grid, binary_grid, sdf, influence_radius, output_path='2d_test_results.png'):
    """
    Visualize the results of the 2D toy example and save to file.
    
    Args:
        points: Nx2 array of coordinates
        labels: N-length array with 1 for foreground, 0 for background
        ground_truth: Binary grid representing the true shape
        prob_grid: Probability grid from extrapolation
        binary_grid: Binary grid after thresholding
        sdf: Signed distance field
        influence_radius: Radius of influence for each point
        output_path: Path to save the visualization image
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot ground truth and points
    axes[0, 0].imshow(ground_truth, cmap='gray')
    fg_points = points[labels == 1]
    bg_points = points[labels == 0]
    axes[0, 0].scatter(fg_points[:, 1], fg_points[:, 0], c='g', s=10, alpha=0.5, label='Foreground')
    axes[0, 0].scatter(bg_points[:, 1], bg_points[:, 0], c='r', s=10, alpha=0.5, label='Background')
    
    # Draw influence radius around a few random foreground points
    for i in range(min(5, len(fg_points))):
        circle = Circle((fg_points[i, 1], fg_points[i, 0]), influence_radius, 
                       fill=False, linestyle='--', color='blue', alpha=0.5)
        axes[0, 0].add_patch(circle)
    
    axes[0, 0].set_title('Ground Truth & Points')
    axes[0, 0].legend()
    
    # Plot probability grid
    im1 = axes[0, 1].imshow(prob_grid, cmap='viridis')
    axes[0, 1].set_title('Probability Grid')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Plot binary grid
    axes[0, 2].imshow(binary_grid, cmap='gray')
    axes[0, 2].set_title('Binary Grid')
    
    # Plot SDF
    im2 = axes[1, 0].imshow(sdf, cmap='coolwarm')
    axes[1, 0].set_title('Signed Distance Field')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Plot zero contour of SDF
    axes[1, 1].imshow(ground_truth, cmap='gray', alpha=0.3)
    contours = axes[1, 1].contour(sdf, levels=[0], colors='r')
    axes[1, 1].set_title('Zero Contour (red) vs Ground Truth')
    
    # Plot hollow grid (used for SDF computation)
    core = binary_erosion(binary_grid, iterations=2)
    hollow_grid = binary_grid & ~core
    axes[1, 2].imshow(hollow_grid, cmap='gray')
    axes[1, 2].set_title('Hollow Structure for SDF')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)  # Close the figure to avoid displaying it

def main():
    # Set parameters
    grid_size = 100
    num_points = 200
    influence_radius = 5.0
    threshold = 0.2
    closing_kernel_size = 3
    smooth_sigma = 1.0
    min_component_size = 10
    
    # Create toy dataset
    points, labels, ground_truth = create_2d_toy_dataset(
        grid_size=grid_size, 
        num_points=num_points, 
        shape_type='ribs',
        noise_level=0.1
    )
    
    # Process the point cloud
    sdf, binary_grid, prob_grid = process_2d_point_cloud(
        points, labels, 
        grid_shape=(grid_size, grid_size),
        influence_radius=influence_radius,
        closing_kernel_size=closing_kernel_size,
        threshold=threshold,
        smooth_sigma=smooth_sigma,
        min_component_size=min_component_size
    )

    visualize_results(points, labels, ground_truth, prob_grid, binary_grid, sdf, influence_radius)
    print(f"SDF shape: {sdf.shape}")
    print(f"Min SDF value: {np.min(sdf)}, Max SDF value: {np.max(sdf)}")
    print(f"Binary grid: {np.sum(binary_grid)} foreground pixels out of {binary_grid.size}")
    print(f"MSE between ground truth and binary grid: {np.mean((ground_truth.astype(int) - binary_grid.astype(int))**2)}")
    
    # Compare with using the original 3D function but with mock 3D data
    print("\nTesting with original 3D function...")
    # Add a third dimension to the points (z=0)
    points_3d = np.hstack([points, np.zeros((len(points), 1))])
    
    # Test with the original function by creating a 1-slice 3D volume
    grid_shape_3d = (grid_size, grid_size, 1)
    try:
        prob_volume_3d = create_extrapolated_volume(points_3d, labels, grid_shape_3d, influence_radius)
        
        # Extract the 2D slice and compare 2D and 3D function results
        prob_grid_3d = prob_volume_3d[:, :, 0]
        mse = np.mean((prob_grid - prob_grid_3d)**2)
        print(f"MSE between 2D and 3D implementation: {mse}")
        if mse < 1e-10:
            print("The 2D and 3D implementations are equivalent!")
        else:
            print("There are some differences between the 2D and 3D implementations.")
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        im1 = axes[0].imshow(prob_grid, cmap='viridis')
        axes[0].set_title('2D Implementation')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(prob_grid_3d, cmap='viridis')
        axes[1].set_title('3D Implementation (z=0 slice)')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig('2d_vs_3d_comparison.png')
        plt.close(fig)  # Close the figure to avoid displaying it
        
    except Exception as e:
        print(f"Error testing 3D function: {str(e)}")
        print("Skipping comparison with 3D implementation.")

if __name__ == "__main__":
    main()