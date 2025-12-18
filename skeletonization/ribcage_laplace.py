from pc_skeletor import LBC
import open3d as o3d

# Load your ribcage (I see it's already in the repo!)
pcd = o3d.io.read_point_cloud("RibFrac1-rib-seg_mesh.ply")

# Extract skeleton with Laplacian-Based Contraction
lbc = LBC(point_cloud=pcd, 
          down_sample=0.01,        # Downsample for faster processing
          init_contraction=1.0,    # Initial contraction strength
          init_attraction=0.5,     # Initial attraction to original positions
          max_iteration_steps=10)  # Limit iterations

# Run the algorithm
contracted_points = lbc.extract_skeleton()
topology = lbc.extract_topology()

# Visualize results
# lbc.visualize()

# Save results
lbc.export_results('./ribcage_skeleton_output')