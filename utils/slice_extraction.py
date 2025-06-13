import open3d as o3d
import numpy as np

# Load point cloud
pcd = o3d.io.read_point_cloud("sparse_occ_200001.0.xyz")

# Debug: Check if file loaded successfully
print(f"Loaded {len(pcd.points)} points")

if len(pcd.points) == 0:
    print("ERROR: No points loaded from XYZ file!")
    print("Open3D might have failed to parse the XYZ file.")
    
    # Try loading with numpy instead
    try:
        points = np.loadtxt("sparse_occ_200001.0.xyz")
        print(f"NumPy loaded {len(points)} points")
        if points.shape[1] >= 3:
            points = points[:, :3]  # Take first 3 columns
        else:
            print(f"File has {points.shape[1]} columns, need at least 3")
    except Exception as e:
        print(f"NumPy loading failed: {e}")
        exit()
else:
    points = np.asarray(pcd.points)

# Debug: Check point cloud bounds
print(f"Point cloud bounds:")
print(f"  X: {np.min(points[:, 0]):.3f} to {np.max(points[:, 0]):.3f}")
print(f"  Y: {np.min(points[:, 1]):.3f} to {np.max(points[:, 1]):.3f}")
print(f"  Z: {np.min(points[:, 2]):.3f} to {np.max(points[:, 2]):.3f}")

# Adjust z_target based on actual data range
z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
z_target = (z_min + z_max) / 2  # Use middle of Z range
tolerance = (z_max - z_min) * 0.03  # 3% of Z range as tolerance

print(f"Using z_target = {z_target:.3f}, tolerance = {tolerance:.3f}")

# Find points within the slice
mask = np.abs(points[:, 2] - z_target) <= tolerance
cross_section_points = points[mask]

print(f"Found {len(cross_section_points)} points in cross-section")

if len(cross_section_points) > 0:
    # Create new point cloud from cross-section
    cross_section_pcd = o3d.geometry.PointCloud()
    cross_section_pcd.points = o3d.utility.Vector3dVector(cross_section_points)
    
    # Save the cross-section
    o3d.io.write_point_cloud("cross_section.ply", cross_section_pcd)
    print("Cross-section saved to cross_section.ply")
    
    # Also save as XYZ for easy inspection
    np.savetxt("cross_section.xyz", cross_section_points, fmt='%.6f')
    print("Cross-section saved to cross_section.xyz")
else:
    print("No points found in cross-section. Try adjusting z_target or tolerance.")