import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import argparse
from pathlib import Path
import pc_processor as pc
    
    
def detect_by_edge_similarity(points, edges):
    """
    Detect endpoints by finding points with small angles between edges (indicating
    edges in similar directions) versus points with angles close to 180° (indicating
    opposite directions).
    """
    endpoint_mask = np.zeros(len(points), dtype=bool)
    angle_threshold = 100  # Threshold for angle similarity (in degrees)
    for i in range(len(points)):
        connected_edges = edges[np.any(edges == i, axis=1)]
        if len(connected_edges) < 2:
            continue
        connected_verts = np.unique(connected_edges[connected_edges != i])
        edge_vectors = points[connected_verts] - points[i]
        edge_vectors = edge_vectors / np.linalg.norm(edge_vectors, axis=1)[:, np.newaxis]
        is_endpoint = True
        for j in range(len(edge_vectors)):
            for k in range(j+1, len(edge_vectors)):
                cos_angle = np.dot(edge_vectors[j], edge_vectors[k])
                if cos_angle < np.cos(np.radians(angle_threshold)):  
                    is_endpoint = False
                    break
            if not is_endpoint:
                break
        endpoint_mask[i] = is_endpoint
    return endpoint_mask

def construct_radius_connectivity(points, radius=None):
    """Construct connectivity graph using radius-based neighbor search."""
    if radius is None:
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=3)
        radius = np.mean(distances[:, 1]) * 2.35
    print (f"Using radius: {radius}")
    
    tree = cKDTree(points)
    pairs = list(tree.query_pairs(radius))
    edges = np.array(pairs) if pairs else np.zeros((0, 2), dtype=int)
    
    return edges, radius

def detect_endpoints_by_density(points, box_size=None):
    """Detect endpoints by counting points within a local box neighborhood."""
    if box_size is None:
        # Compute box size based on point cloud characteristics
        bbox = np.ptp(points, axis=0)  # Get range of points in each dimension
        box_size = np.mean(bbox) * 0.05
    
    tree = cKDTree(points)
    endpoint_mask = np.zeros(len(points), dtype=bool)
    
    # For each point, count neighbors within box_size
    for i in range(len(points)):
        query_point = points[i]
        
        # Define box bounds
        box_min = query_point - box_size/2
        box_max = query_point + box_size/2
        
        # Find points within box
        indices = tree.query_ball_point(query_point, box_size/2)
        n_neighbors = len(indices)
        
        # If fewer than threshold neighbors, mark as endpoint
        if n_neighbors < 6:  # You can adjust this threshold
            endpoint_mask[i] = True
            
    return endpoint_mask

def identify_separate_skeletons(points, edges, n_points):
    """Identify separate skeleton components in the point cloud."""
    adj_matrix = csr_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(n_points, n_points)
    )
    adj_matrix = adj_matrix + adj_matrix.T
    
    n_components, labels = connected_components(
        csgraph=adj_matrix,
        directed=False,
        return_labels=True
    )
    
    return n_components, labels

def find_endpoints_per_skeleton(edges, n_points, component_labels):
    """Find endpoints for each separate skeleton component."""
    connections = np.zeros(n_points)
    for edge in edges:
        connections[edge[0]] += 1
        connections[edge[1]] += 1
    
    component_endpoints = {}
    unique_components = np.unique(component_labels)
    
    for component in unique_components:
        component_mask = (component_labels == component)
        component_connections = connections[component_mask]
        local_endpoints = np.where(component_connections == 1)[0]
        global_endpoints = np.where(component_mask)[0][local_endpoints]
        component_endpoints[component] = global_endpoints
    
    return component_endpoints, connections

def get_component_means(points, component_labels):
    """Calculate mean coordinates for each skeleton component."""
    unique_components = np.unique(component_labels)
    component_means = {}
    
    for component in unique_components:
        component_mask = (component_labels == component)
        component_points = points[component_mask]
        mean_coords = np.mean(component_points, axis=0)
        component_means[component] = mean_coords
        
    return component_means

def group_components_into_ribs(component_means, horizontal_threshold=20, vertical_threshold=10):
    """Group components that likely belong to the same rib based on proximity."""
    # Convert component_means to a list of tuples (component_id, coordinates)
    # Convert numpy integers to regular integers here
    components = [(int(comp_id), coords) for comp_id, coords in component_means.items()]
    
    # Sort components by y-coordinate (vertical position) and x-coordinate
    sorted_components = sorted(components, key=lambda x: (x[1][1], x[1][0]))
    
    # Initialize rib groups
    rib_groups = []
    processed_components = set()
    
    # Group components that are close to each other
    for i, (comp_id, coords) in enumerate(sorted_components):
        if comp_id in processed_components:
            continue
            
        current_group = [comp_id]  # Now using regular integer
        processed_components.add(comp_id)
        
        # Check remaining components for proximity
        for j, (other_id, other_coords) in enumerate(sorted_components[i+1:], i+1):
            if other_id in processed_components:
                continue
                
            # Calculate horizontal and vertical distances
            horizontal_dist = abs(coords[0] - other_coords[0])
            vertical_dist = abs(coords[1] - other_coords[1])
            
            if horizontal_dist < horizontal_threshold and vertical_dist < vertical_threshold:
                current_group.append(other_id)  # Now using regular integer
                processed_components.add(other_id)
        
        rib_groups.append(current_group)
    
    return rib_groups

def assign_rib_numbers(rib_groups, component_means):
    """
    Assign rib numbers enforcing exactly 12 ribs on each side.
    Will merge or split groups as needed to achieve this.
    """
    # Calculate mean position for each rib group
    rib_positions = []
    for i, group in enumerate(rib_groups):
        if not group:  # Skip empty groups
            continue
        group_coords = np.mean([component_means[comp_id] for comp_id in group], axis=0)
        rib_positions.append((i, group_coords))
    
    if not rib_positions:  # Handle case where all groups are empty
        return {}
    
    # Split into left and right sides based on x-coordinate
    median_x = np.median([pos[1][0] for pos in rib_positions])
    left_positions = [(idx, pos) for idx, pos in rib_positions if pos[0] < median_x]
    right_positions = [(idx, pos) for idx, pos in rib_positions if pos[0] >= median_x]
    
    def adjust_to_12_ribs(positions, side):
        """Adjust number of groups to exactly 12 ribs."""
        # Sort by Y coordinate
        sorted_positions = sorted(positions, key=lambda x: x[1][1])
        
        if len(sorted_positions) > 12:
            # If we have too many groups, merge closest ones
            while len(sorted_positions) > 12:
                # Find closest pair by Y coordinate
                min_dist = float('inf')
                merge_idx = 0
                for i in range(len(sorted_positions) - 1):
                    dist = abs(sorted_positions[i][1][1] - sorted_positions[i + 1][1][1])
                    if dist < min_dist:
                        min_dist = dist
                        merge_idx = i
                
                # Merge the groups
                group1_idx = sorted_positions[merge_idx][0]
                group2_idx = sorted_positions[merge_idx + 1][0]
                rib_groups[group1_idx].extend(rib_groups[group2_idx])
                # Update mean position
                new_mean = np.mean([component_means[comp_id] for comp_id in rib_groups[group1_idx]], axis=0)
                
                # Remove the merged group and update positions
                sorted_positions.pop(merge_idx + 1)
                sorted_positions[merge_idx] = (group1_idx, new_mean)
        
        elif len(sorted_positions) < 12:
            # If we have too few groups, split largest gaps
            sorted_positions = sorted(sorted_positions, key=lambda x: x[1][1])
            while len(sorted_positions) < 12:
                # Find largest gap in Y coordinates
                max_gap = 0
                split_idx = 0
                for i in range(len(sorted_positions) - 1):
                    gap = abs(sorted_positions[i + 1][1][1] - sorted_positions[i][1][1])
                    if gap > max_gap:
                        max_gap = gap
                        split_idx = i
                
                # Create a new empty group at the midpoint
                mid_y = (sorted_positions[split_idx][1][1] + sorted_positions[split_idx + 1][1][1]) / 2
                mid_x = sorted_positions[split_idx][1][0]  # Keep same x coordinate
                mid_z = (sorted_positions[split_idx][1][2] + sorted_positions[split_idx + 1][1][2]) / 2
                new_pos = np.array([mid_x, mid_y, mid_z])
                
                # Add new empty group
                rib_groups.append([])
                new_group_idx = len(rib_groups) - 1
                sorted_positions.insert(split_idx + 1, (new_group_idx, new_pos))
        
        return sorted_positions
    
    # Adjust both sides to exactly 12 ribs
    left_positions = adjust_to_12_ribs(left_positions, 'left')
    right_positions = adjust_to_12_ribs(right_positions, 'right')
    
    # Assign rib numbers
    rib_numbers = {}
    
    # Left side: 1-12 from top to bottom
    for i, (group_idx, _) in enumerate(sorted(left_positions, key=lambda x: x[1][1])):
        rib_num = i + 1
        for comp_id in rib_groups[group_idx]:
            rib_numbers[int(comp_id)] = f"L{rib_num}"
    
    # Right side: 1-12 from top to bottom (changed from previous bottom-to-top)
    for i, (group_idx, _) in enumerate(sorted(right_positions, key=lambda x: x[1][1])):
        rib_num = i + 1
        for comp_id in rib_groups[group_idx]:
            rib_numbers[int(comp_id)] = f"R{rib_num}"
    
    return rib_numbers

def analyze_and_visualize_skeleton(input_file, output_obj, radius=None):
    points = pc.load_skeleton_file(input_file)
    edges, used_radius = construct_radius_connectivity(points, radius)
    
    n_components, component_labels = identify_separate_skeletons(points, edges, len(points))
    
    # Get component means
    component_means = get_component_means(points, component_labels)
    
    # Group components into ribs
    rib_groups = group_components_into_ribs(component_means)
    
    # Assign rib numbers
    rib_numbers = assign_rib_numbers(rib_groups, component_means)
    
    # Get endpoints from all methods
    connectivity_endpoints, connections = find_endpoints_per_skeleton(edges, len(points), component_labels)
    density_endpoints = detect_endpoints_by_density(points)
    edge_similarity_endpoints = detect_by_edge_similarity(points, edges)
    
    # Track detection counts
    method_counts = {
        'connectivity': 0,
        'density': 0,
        'edge_similarity': 0
    }
    
    combined_endpoints = {}
    for component in range(n_components):
        component_mask = (component_labels == component)
        component_points = np.where(component_mask)[0]
        
        endpoints_with_method = []
        processed_points = set()
        
        methods = [
            ('connectivity', connectivity_endpoints.get(component, [])),
            ('density', component_points[density_endpoints[component_mask]]),
            ('edge_similarity', component_points[edge_similarity_endpoints[component_mask]])
        ]
        
        for method_name, method_endpoints in methods:
            for endpoint in method_endpoints:
                if endpoint not in processed_points:
                    endpoints_with_method.append((endpoint, method_name))
                    processed_points.add(endpoint)
                method_counts[method_name] += 1
        
        combined_endpoints[component] = endpoints_with_method
    
        pc.create_visualization_obj(points, edges, combined_endpoints, output_obj)
    
    # Sort components by rib number
    def get_rib_sort_key(item):
        component, rib_id = item
        side = rib_id[0]  # 'R' or 'L'
        number = int(rib_id[1:])  # Rib number
        # Sort R1-R12 first, then L1-L12
        return (0 if side == 'R' else 1, number, component)
    
    sorted_components = sorted([(comp, rib_numbers[comp]) for comp in range(n_components)], 
                             key=get_rib_sort_key)
    
    # Print sorted component information
    print("\nComponent Analysis:")
    for component in range(n_components):
        mean_coords = component_means[component]
        rib_id = rib_numbers.get(component, "Unassigned")
        print(f"Component {component} (Rib {rib_id}):")
        print(f"  Mean coordinates: X={mean_coords[0]:.2f}, Y={mean_coords[1]:.2f}, Z={mean_coords[2]:.2f}")
    
    # Print sorted rib grouping information
    print("\nRib Groups:")
    # Create a dictionary to group components by rib ID
    rib_to_components = {}
    for group in rib_groups:
        if not group:  # Skip empty groups
            continue
        rib_id = rib_numbers.get(group[0], "Unassigned")
        rib_to_components[rib_id] = group
    
    # Create list of all possible rib IDs
    all_rib_ids = ([f"R{i}" for i in range(1, 13)] + 
                   [f"L{i}" for i in range(1, 13)])
    
    # Print all ribs in order
    for rib_id in all_rib_ids:
        if rib_id in rib_to_components:
            group = rib_to_components[rib_id]
            print(f"Rib {rib_id}: Components {sorted(group)}")
            if len(group) > 1:
                print("  Potential fracture detected (multiple components)")
        else:
            print(f"Rib {rib_id}: No components (empty)")
    
    return n_components, combined_endpoints, rib_numbers, rib_groups

def process_directory(directory_path, radius=None):
    """Process all OBJ and XYZ files in the given directory."""
    directory = Path(directory_path)
    results = {}
    
    output_dir = directory / "endpoint_analysis"
    output_dir.mkdir(exist_ok=True)
    
    skeleton_files = list(directory.glob("*.obj")) + list(directory.glob("*.xyz"))
    if not skeleton_files:
        print(f"No OBJ or XYZ files found in {directory}")
        return
    
    print(f"Found {len(skeleton_files)} files to process")
    
    for file in skeleton_files:
        if file.stem.endswith("_endpoints"):
            continue
        print(f"\nProcessing: {file.name}")
        output_file = output_dir / f"{file.stem}_endpoints.ply"
        
        try:
            n_components, component_endpoints, rib_numbers, rib_groups = analyze_and_visualize_skeleton(
                str(file), str(output_file), radius
            )
            
            # Convert numpy integers to regular integers
            converted_rib_groups = []
            for group in rib_groups:
                if group:  # Only process non-empty groups
                    converted_group = [int(comp) for comp in group]
                    converted_rib_groups.append(converted_group)
                else:
                    converted_rib_groups.append([])  # Keep empty groups as empty lists
            
            # Count potential fractures (groups with multiple components)
            fracture_count = sum(1 for group in converted_rib_groups if len(group) > 1)
            
            results[file.name] = {
                'n_components': n_components,
                'endpoints_per_component': {int(comp): len(endpoints) 
                                         for comp, endpoints in component_endpoints.items()},
                'rib_assignments': {int(comp): rib for comp, rib in rib_numbers.items()},
                'potential_fractures': converted_rib_groups,
                'fracture_count': fracture_count  # Add fracture count to results
            }
            
        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")
            continue
    
    # Write summary report
    report_path = output_dir / "analysis_summary.txt"
    with open(report_path, 'w') as f:
        f.write("Skeleton Analysis Summary\n")
        f.write("=======================\n\n")
        for filename, data in results.items():
            f.write(f"File: {filename}\n")
            f.write(f"Number of skeletons: {data['n_components']}\n")
            f.write(f"Number of potential fractures: {data['fracture_count']}\n")  # Add fracture count
            f.write("\nRib Assignments:\n")
            
            # Create lists of all ribs to ensure we show even empty ones
            all_ribs = ([f"R{i}" for i in range(1, 13)] + 
                       [f"L{i}" for i in range(1, 13)])
            
            # Track which components belong to each rib
            rib_to_comps = {}
            for comp, rib in data['rib_assignments'].items():
                if rib not in rib_to_comps:
                    rib_to_comps[rib] = []
                rib_to_comps[rib].append(comp)
            
            # Print all ribs in order, indicating empty ones
            for rib_id in all_ribs:
                if rib_id in rib_to_comps:
                    components = sorted(rib_to_comps[rib_id])
                    f.write(f"  Rib {rib_id}: Components {components}\n")
                else:
                    f.write(f"  Rib {rib_id}: No components (empty)\n")
            
            if data['potential_fractures']:
                f.write("\nPotential Fractures:\n")
                for group in data['potential_fractures']:
                    if group and len(group) > 1:  # Only print groups with multiple components
                        rib = data['rib_assignments'].get(group[0], "Unassigned")
                        f.write(f"  Rib {rib}: Components {sorted(group)}\n")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Perform geometric analysis on OBJ files in a directory")
    parser.add_argument("directory_path", help="Path to the directory containing OBJ files")
    parser.add_argument("--radius", type=float, help="Optional: Specify connectivity radius", default=None)
    parser.add_argument("--horizontal-threshold", type=float, help="Threshold for horizontal component grouping", default=20)
    parser.add_argument("--vertical-threshold", type=float, help="Threshold for vertical component grouping", default=10)
    args = parser.parse_args()
    process_directory(args.directory_path, args.radius)

if __name__ == "__main__":
    main()