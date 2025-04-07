import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import kimimaro
from scipy import ndimage
import os
import warnings
from scipy.spatial import cKDTree
import scipy.sparse as sparse
from scipy.sparse.csgraph import minimum_spanning_tree

def extract_rib_skeleton_kimimaro(points, labels, voxel_size=1.0, debug=True):  # Reduced voxel size to 1.0
    """
    Extract rib cage skeleton using kimimaro, processing each component separately
    with anatomical constraints and improved connection logic
    """
    # Convert point cloud to volumetric data
    print("Converting point cloud to volume...")
    
    # Ensure points array is float64 to prevent casting issues
    points = points.astype(np.float64)
    
    # Find bounds of the point cloud
    min_coords = np.min(points, axis=0).astype(np.float64)
    max_coords = np.max(points, axis=0).astype(np.float64)
    
    # Add padding to ensure the object doesn't touch the volume boundary
    padding = 5 * float(voxel_size)
    min_coords -= padding
    max_coords += padding
    
    # Compute dimensions
    dimensions = np.ceil((max_coords - min_coords) / float(voxel_size)).astype(np.int32)
    volume = np.zeros(dimensions, dtype=np.uint8)
    
    # Fill volume with labeled points
    rib_points = points[labels == 1]
    n_filled_voxels = 0
    
    for point in rib_points:
        idx = tuple(np.floor((point - min_coords) / float(voxel_size)).astype(np.int32))
        if all(0 <= idx[i] < dimensions[i] for i in range(3)):
            volume[idx] = 1
            n_filled_voxels += 1
    
    print(f"Created volume of shape {volume.shape} with {n_filled_voxels} filled voxels")
    
    # Apply slight dilation to connect very close points
    struct = ndimage.generate_binary_structure(3, 1)  # 6-connected
    dilated = ndimage.binary_dilation(volume, structure=struct, iterations=1)
    
    # Label connected components
    labeled_volume, num_features = ndimage.label(dilated)
    component_sizes = np.bincount(labeled_volume.ravel())[1:]  # Skip background
    print(f"Found {num_features} connected components")
    
    if len(component_sizes) == 0:
        raise ValueError("No connected components found after dilation")
    
    # Process each significant component
    # Sort components by size in descending order
    sorted_indices = np.argsort(-component_sizes)
    
    # Process components above a minimum size rather than just top N
    min_component_size = 10  # Minimum size in voxels
    valid_indices = [i for i in sorted_indices if component_sizes[i] >= min_component_size]
    print(f"Processing {len(valid_indices)} components with at least {min_component_size} voxels")
    
    all_skeletons = []
    
    # Process each valid component
    for i, comp_idx_offset in enumerate(valid_indices):
        comp_idx = comp_idx_offset + 1  # Add 1 because 0 is background
        comp_size = component_sizes[comp_idx_offset]
            
        print(f"Processing component {i+1}/{len(valid_indices)} (size: {comp_size} voxels)...")
        
        # Extract this component
        comp_volume = (labeled_volume == comp_idx).astype(np.uint8)
        
        # Configure kimimaro parameters
        teasar_params = {
            'scale': 5,
            'const': 100,
            'pdrf_exponent': 4,  # Integer
            'pdrf_scale': 100000,
        }
        
        # Skeletonize this component
        try:
            comp_skeletons = kimimaro.skeletonize(
                comp_volume,
                teasar_params=teasar_params,
                dust_threshold=1,  # Keep even tiny components
                anisotropy=(1.0, 1.0, 1.0),
                fix_branching=True,
                fix_borders=True,
                progress=True,
                parallel=1,
            )
            
            if comp_skeletons:
                comp_key = list(comp_skeletons.keys())[0]
                comp_skel = comp_skeletons[comp_key]
                
                # Only add if it has vertices
                if len(comp_skel.vertices) > 0:
                    # Remap vertices back to original coordinate space
                    vertices = comp_skel.vertices.astype(np.float64)
                    voxel_size_float = float(voxel_size)
                    comp_skel.vertices = vertices * voxel_size_float + min_coords
                    
                    all_skeletons.append(comp_skel)
                    print(f"  Added skeleton with {len(comp_skel.vertices)} vertices, {len(comp_skel.edges)} edges")
            
        except Exception as e:
            print(f"  Error skeletonizing component {i+1}: {str(e)}")
            # Continue with next component
    
    if not all_skeletons:
        raise ValueError("Failed to extract any skeletons from components")
    
    # Combine all skeletons
    print(f"Combining {len(all_skeletons)} skeletons...")
    from cloudvolume import PrecomputedSkeleton
    
    combined_vertices = []
    combined_edges = []
    vertex_offset = 0
    
    for skel in all_skeletons:
        # Add vertices
        combined_vertices.append(skel.vertices)
        
        # Adjust edges for the offset and add them
        if len(skel.edges) > 0:
            adjusted_edges = skel.edges.copy() + vertex_offset
            combined_edges.append(adjusted_edges)
        
        vertex_offset += len(skel.vertices)
    
    # Create the combined skeleton
    combined_vertices = np.vstack(combined_vertices)
    combined_edges = np.vstack(combined_edges) if combined_edges else np.array([])
    combined_skel = PrecomputedSkeleton(vertices=combined_vertices, edges=combined_edges)
    
    # Connect disconnected parts with improved anatomical constraints
    result_skel = connect_skeleton_parts(combined_skel, max_distance=25.0)  # Reduced from 50.0
    
    # Post-process the skeleton: smooth and prune small branches
    result_skel = post_process_skeleton(result_skel)
    
    print(f"Final skeleton has {len(result_skel.vertices)} vertices and {len(result_skel.edges)} edges")
    return result_skel

def connect_skeleton_parts(skel, max_distance=25.0):
    """
    Connect disconnected parts of the skeleton using a minimum spanning tree
    with improved anatomical constraints
    
    Args:
        skel: Skeleton object with vertices and edges
        max_distance: Maximum distance to connect disconnected parts
        
    Returns:
        Connected skeleton
    """
    from cloudvolume import PrecomputedSkeleton
    
    # If we already have enough edges or no vertices, return as is
    if len(skel.vertices) <= 1 or len(skel.edges) >= len(skel.vertices) - 1:
        return skel
    
    # Find connected components in the current skeleton
    # Build an adjacency matrix from the edges
    n = len(skel.vertices)
    adj_matrix = sparse.lil_matrix((n, n), dtype=bool)
    
    for e in skel.edges:
        if e[0] < n and e[1] < n:  # Safety check
            adj_matrix[e[0], e[1]] = True
            adj_matrix[e[1], e[0]] = True
    
    # Use breadth-first search to find connected components
    visited = np.zeros(n, dtype=bool)
    components = []
    
    for i in range(n):
        if not visited[i]:
            component = []
            queue = [i]
            visited[i] = True
            
            while queue:
                node = queue.pop(0)
                component.append(node)
                
                neighbors = adj_matrix[node].nonzero()[1]
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            components.append(component)
    
    print(f"Found {len(components)} disconnected components in skeleton")
    
    if len(components) <= 1:
        return skel
    
    # For each pair of components, find the closest vertices
    new_edges = list(skel.edges)
    component_pairs = []
    
    # Create a KD-tree for each component to speed up closest point queries
    kdtrees = []
    for comp in components:
        comp_vertices = skel.vertices[comp]
        kdtrees.append((cKDTree(comp_vertices), comp))
    
    # For each pair of components, find closest vertices with anatomical constraints
    for i in range(len(components)):
        tree_i, comp_i = kdtrees[i]
        
        for j in range(i+1, len(components)):
            tree_j, comp_j = kdtrees[j]
            
            # Query the closest pair between the two components
            distances, indices = tree_i.query(skel.vertices[comp_j], k=1)
            
            if isinstance(distances, np.ndarray):
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                i_closest = indices[min_idx] if isinstance(indices, np.ndarray) else indices
                j_closest = min_idx
            else:
                min_dist = distances
                i_closest = indices
                j_closest = 0
            
            # Only connect if within max distance and satisfies anatomical constraints
            if min_dist <= max_distance:
                idx_i = comp_i[i_closest]
                idx_j = comp_j[j_closest]
                
                # Anatomical constraint: Avoid connecting points with large vertical differences
                # This helps prevent incorrect connections between different ribs
                vertical_diff = abs(skel.vertices[idx_i][2] - skel.vertices[idx_j][2])
                max_vertical_diff = min(max_distance * 0.5, 15.0)  # Limit vertical connections
                
                if vertical_diff <= max_vertical_diff:
                    component_pairs.append((idx_i, idx_j, min_dist))
                
    # Sort pairs by distance
    component_pairs.sort(key=lambda x: x[2])
    
    # Use Kruskal's algorithm to construct a minimum spanning tree connecting components
    # First initialize each component as its own set
    parent = {i: i for i in range(len(components))}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        parent[find(x)] = find(y)
    
    # Find component index for a vertex
    vertex_to_component = {}
    for i, comp in enumerate(components):
        for v in comp:
            vertex_to_component[v] = i
    
    # Add edges to connect components (minimum spanning tree between components)
    for u, v, dist in component_pairs:
        comp_u = vertex_to_component[u]
        comp_v = vertex_to_component[v]
        
        if find(comp_u) != find(comp_v):
            new_edges.append((u, v))
            union(comp_u, comp_v)
    
    # Create a new skeleton with the updated edges
    new_skel = PrecomputedSkeleton(
        vertices=skel.vertices,
        edges=np.array(new_edges)
    )
    
    print(f"Added {len(new_edges) - len(skel.edges)} new connections between components")
    return new_skel

def post_process_skeleton(skel, smoothing_factor=0.2):
    """
    Post-process the skeleton by smoothing and pruning small branches
    
    Args:
        skel: Skeleton object with vertices and edges
        smoothing_factor: Amount of smoothing to apply (0.0-1.0)
        
    Returns:
        Processed skeleton
    """
    from cloudvolume import PrecomputedSkeleton
    
    # If there are no vertices or edges, return as is
    if len(skel.vertices) == 0 or len(skel.edges) == 0:
        return skel
    
    # Create an adjacency list
    n = len(skel.vertices)
    adjacency = [[] for _ in range(n)]
    
    for edge in skel.edges:
        if edge[0] < n and edge[1] < n:  # Safety check
            adjacency[edge[0]].append(edge[1])
            adjacency[edge[1]].append(edge[0])
    
    # Identify branch points and endpoints
    branch_points = []
    endpoints = []
    
    for i in range(n):
        if len(adjacency[i]) > 2:
            branch_points.append(i)
        elif len(adjacency[i]) == 1:
            endpoints.append(i)
    
    # Prune very small branches
    min_branch_length = 3  # Minimum number of edges in a branch
    branches_to_remove = []
    
    for endpoint in endpoints:
        current = endpoint
        branch = [current]
        prev = -1
        
        # Follow branch until we reach a branch point
        while len(adjacency[current]) <= 2:
            # Find next vertex that is not the previous one
            neighbors = adjacency[current]
            if len(neighbors) == 0:
                break
                
            if len(neighbors) == 1:
                # This is an endpoint
                next_vert = neighbors[0]
            else:  # len(neighbors) == 2
                # This is a path vertex
                next_vert = neighbors[0] if neighbors[0] != prev else neighbors[1]
            
            prev = current
            current = next_vert
            branch.append(current)
            
            if current in branch_points or len(branch) > min_branch_length:
                # We've reached a branch point or exceeded minimum length
                break
        
        # If branch is too short and doesn't reach a branch point, mark for removal
        if len(branch) <= min_branch_length and current not in branch_points:
            branches_to_remove.extend(branch)
    
    # Create a new vertex array without the removed branches
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[branches_to_remove] = False
    
    # Create a mapping from old indices to new indices
    new_indices = np.cumsum(keep_mask) - 1
    
    # Create new edges list
    new_edges = []
    for u, v in skel.edges:
        if keep_mask[u] and keep_mask[v]:
            new_edges.append((new_indices[u], new_indices[v]))
    
    # Create new vertex array
    new_vertices = skel.vertices[keep_mask]
    
    # Smooth the vertices
    # For each vertex that's not an endpoint or branch point, adjust its position
    # based on its neighbors
    if smoothing_factor > 0:
        smoothed_vertices = new_vertices.copy()
        
        # Recreate adjacency list for the pruned skeleton
        n_new = len(new_vertices)
        new_adjacency = [[] for _ in range(n_new)]
        
        for u, v in new_edges:
            new_adjacency[u].append(v)
            new_adjacency[v].append(u)
        
        # Identify new branch points and endpoints
        new_branch_points = set()
        new_endpoints = set()
        
        for i in range(n_new):
            if len(new_adjacency[i]) > 2:
                new_branch_points.add(i)
            elif len(new_adjacency[i]) == 1:
                new_endpoints.add(i)
        
        # Smooth internal vertices
        for i in range(n_new):
            if i not in new_branch_points and i not in new_endpoints and new_adjacency[i]:
                # Calculate average position of neighbors
                neighbor_pos = np.mean([new_vertices[j] for j in new_adjacency[i]], axis=0)
                # Blend original position with neighbor average
                smoothed_vertices[i] = (1 - smoothing_factor) * new_vertices[i] + smoothing_factor * neighbor_pos
        
        new_vertices = smoothed_vertices
    
    # Create new skeleton with pruned and smoothed vertices
    new_skel = PrecomputedSkeleton(
        vertices=new_vertices,
        edges=np.array(new_edges)
    )
    
    print(f"Post-processing: removed {n - len(new_vertices)} vertices and smoothed the skeleton")
    return new_skel

def visualize_kimimaro_skeleton(skel, original_points=None, original_labels=None):
    """
    Visualize the skeleton produced by kimimaro
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original points if provided
    if original_points is not None and original_labels is not None:
        # Subsample for better visualization
        max_points = 3000
        if len(original_points) > max_points:
            idx = np.random.choice(len(original_points), max_points, replace=False)
            points_subset = original_points[idx]
            labels_subset = original_labels[idx]
            mask = labels_subset == 1
        else:
            points_subset = original_points
            mask = original_labels == 1
            
        ax.scatter(points_subset[mask, 0], points_subset[mask, 1], points_subset[mask, 2],
                  c='lightgray', s=1, alpha=0.2, label='Original Rib Points')
    
    # Plot vertices
    ax.scatter(skel.vertices[:, 0], skel.vertices[:, 1], skel.vertices[:, 2],
               c='red', s=10, label='Skeleton Vertices')
    
    # Plot edges
    if len(skel.edges) > 0:
        for edge in skel.edges:
            if edge[0] < len(skel.vertices) and edge[1] < len(skel.vertices):  # Safety check
                p1 = skel.vertices[edge[0]]
                p2 = skel.vertices[edge[1]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'r-', linewidth=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Rib Cage Skeleton')
    
    plt.tight_layout()
    plt.show()

def extract_rib_cage_skeleton(points_file, labels_file, voxel_size=1.0):  # Reduced voxel size to 1.0
    """
    Extract skeleton from rib cage point cloud using kimimaro with improved preprocessing
    """
    # Load point cloud and labels
    points = np.load(points_file)
    labels = np.load(labels_file)
    
    # Ensure consistent data types
    points = points.astype(np.float64)
    labels = labels.astype(np.int32)
    voxel_size = float(voxel_size)
    
    print(f"Loaded point cloud with {len(points)} points")
    print(f"Number of rib cage points: {np.sum(labels == 1)}")
    
    # Use kimimaro for skeletonization with multi-component approach
    print("Using kimimaro with improved multi-component approach...")
    skel = extract_rib_skeleton_kimimaro(points, labels, voxel_size=voxel_size)
    visualize_kimimaro_skeleton(skel, points, labels)
    
    return skel

def visualize_point_cloud(points, labels):
    """
    Visualize the original point cloud to verify data
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    mask = labels == 1
    ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
              c='blue', s=1, alpha=0.5, label='Rib Points')
    
    # Plot non-rib points if any
    if np.any(~mask):
        ax.scatter(points[~mask, 0], points[~mask, 1], points[~mask, 2],
                  c='gray', s=1, alpha=0.2, label='Non-Rib Points')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    points_file = "/path/to/point/testRibFrac504.npy"
    labels_file = "/path/to/label/testRibFrac504.npy"
    
    # Use kimimaro for skeleton extraction with improved multi-component approach
    try:
        result = extract_rib_cage_skeleton(
            points_file, 
            labels_file,
            voxel_size=1.0  # Reduced voxel size for better detail
        )
        
        print(f"Extracted skeleton with {len(result.vertices)} vertices and {len(result.edges)} edges")
    except Exception as e:
        print(f"Failed to extract skeleton: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Check your input data and try adjusting parameters.")