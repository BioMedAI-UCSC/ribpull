import argparse
from pathlib import Path
import csv
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np
import pc_processor as pc
import rib_segmentation as rs

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

def identify_potential_fractures(points, combined_endpoints, xy_distance_threshold=10, z_distance_threshold=5):
    """
    Identify endpoints using both XY Euclidean distance and Z distance thresholds.
    Keeps endpoints that have any other endpoint within both distance thresholds.
    
    Parameters:
    points (np.array): The point cloud data
    combined_endpoints (dict): Dictionary mapping component IDs to endpoint indices with detection methods
    xy_distance_threshold (float): Maximum Euclidean distance in XY plane for endpoints to be considered neighbors
    z_distance_threshold (float): Maximum distance in Z axis for endpoints to be considered neighbors
    
    Returns:
    dict: Dictionary of filtered endpoints
    """
    import numpy as np
    from scipy.spatial import cKDTree
    
    # Create a dictionary to store filtered endpoints
    filtered_endpoints = {}
    for component_id in combined_endpoints:
        filtered_endpoints[component_id] = []
    
    # Extract all endpoint coordinates and their metadata
    all_endpoints_data = []
    for component_id, endpoints_with_method in combined_endpoints.items():
        for endpoint_index, detection_method in endpoints_with_method:
            coords = points[endpoint_index]
            all_endpoints_data.append({
                'component_id': component_id,
                'endpoint_index': endpoint_index,
                'detection_method': detection_method,
                'coords': coords
            })
    
    if not all_endpoints_data:
        return filtered_endpoints
    
    # Create a KDTree for efficient spatial queries (using ONLY X,Y coordinates)
    endpoint_coords = np.array([ep['coords'] for ep in all_endpoints_data])
    endpoint_xy_coords = endpoint_coords[:, :2]  # Extract only X and Y coordinates
    tree = cKDTree(endpoint_xy_coords)
    
    # Track which endpoints to keep
    endpoints_to_keep = set()
    
    # Store pairs of neighboring endpoints for reporting
    neighbor_pairs = []
    
    # Process each endpoint
    for i, endpoint_data in enumerate(all_endpoints_data):
        # Find all points within the XY distance threshold
        # Returns indices of all points within xy_distance_threshold (including self)
        xy_neighbors = tree.query_ball_point(endpoint_xy_coords[i], xy_distance_threshold)
        
        # Filter neighbors by Z distance
        valid_neighbors = []
        for neighbor_idx in xy_neighbors:
            if neighbor_idx == i:  # Skip self
                continue
                
            # Check Z distance
            z_distance = abs(endpoint_coords[i][2] - endpoint_coords[neighbor_idx][2])
            if z_distance <= z_distance_threshold:
                valid_neighbors.append(neighbor_idx)
        
        # If this endpoint has any valid neighbors (close in both XY and Z), keep it
        if valid_neighbors:
            endpoints_to_keep.add(i)
            
            # Record neighbor pairs for reporting
            for neighbor_idx in valid_neighbors:
                pair = (min(i, neighbor_idx), max(i, neighbor_idx))  # Sort to avoid duplicates
                if pair not in neighbor_pairs:
                    neighbor_pairs.append(pair)
                    endpoints_to_keep.add(neighbor_idx)  # Also keep the neighbor
    
    # Build the filtered endpoints dictionary
    for i in endpoints_to_keep:
        endpoint_data = all_endpoints_data[i]
        component_id = endpoint_data['component_id']
        endpoint_index = endpoint_data['endpoint_index']
        detection_method = endpoint_data['detection_method']
        
        filtered_endpoints[component_id].append((endpoint_index, detection_method))
    
    # Print information about filtered endpoints
    total_endpoints = sum(len(endpoints) for endpoints in combined_endpoints.values())
    kept_endpoints = sum(len(endpoints) for endpoints in filtered_endpoints.values())
    
    print(f"Found {len(neighbor_pairs)} neighbor pairs involving {kept_endpoints} endpoints")
    print(f"({kept_endpoints} out of {total_endpoints} total endpoints, {kept_endpoints/total_endpoints*100:.1f}%)")
    print(f"Using XY threshold: {xy_distance_threshold}, Z threshold: {z_distance_threshold}")
    
    # Print details about a few sample neighbor pairs
    print("\nSample neighbor pairs (up to 5):")
    for idx, pair in enumerate(neighbor_pairs[:5]):  # Limit to first 5 pairs to avoid excessive output
        ep1 = all_endpoints_data[pair[0]]
        ep2 = all_endpoints_data[pair[1]]
        coords1 = ep1['coords']
        coords2 = ep2['coords']
        # Calculate distances
        xy_dist = np.linalg.norm(coords1[:2] - coords2[:2])
        z_dist = abs(coords1[2] - coords2[2])
        
        print(f"  Pair {idx+1}:")
        print(f"    Endpoint 1: Component {ep1['component_id']}, "
              f"XYZ: ({coords1[0]:.2f}, {coords1[1]:.2f}, {coords1[2]:.2f})")
        print(f"    Endpoint 2: Component {ep2['component_id']}, "
              f"XYZ: ({coords2[0]:.2f}, {coords2[1]:.2f}, {coords2[2]:.2f})")
        print(f"    XY Distance: {xy_dist:.2f}, Z Distance: {z_dist:.2f}")
    
    if len(neighbor_pairs) > 5:
        print(f"  ... and {len(neighbor_pairs) - 5} more pairs")
    
    return filtered_endpoints

def construct_radius_connectivity(points, radius=None):
    """Construct connectivity graph using radius-based neighbor search."""
    if radius is None:
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=3)
        radius = np.mean(distances[:, 1]) * 2.35
    print(f"Using radius: {radius}")
    
    tree = cKDTree(points)
    pairs = list(tree.query_pairs(radius))
    edges = np.array(pairs) if pairs else np.zeros((0, 2), dtype=int)
    
    return edges, radius

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

def analyze_and_visualize_skeleton(input_file, output_obj, radius=None):
    points = pc.load_skeleton_file(input_file)
    edges, used_radius = construct_radius_connectivity(points, radius)
    
    # From components to rib labels
    n_components, component_labels = identify_separate_skeletons(points, edges, len(points))
    component_means = get_component_means(points, component_labels)
    rib_groups = rs.group_components_into_ribs(component_means)
    rib_numbers = rs.assign_rib_numbers(rib_groups, component_means)
    
    # Get endpoints from all methods
    connectivity_endpoints, connections = find_endpoints_per_skeleton(edges, len(points), component_labels)
    density_endpoints = detect_endpoints_by_density(points)
    edge_similarity_endpoints = detect_by_edge_similarity(points, edges)
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

    potential_fracture_endpoints = identify_potential_fractures(points, combined_endpoints)
    
    # Create visualization file - MODIFY THIS TO USE potential_fracture_endpoints
    pc.create_visualization_obj(points, edges, potential_fracture_endpoints, output_obj)
    
    # Create CSV output filename from output_obj path
    output_csv = output_obj.rsplit('.', 1)[0] + '_coordinates.csv'
    
    # Export endpoints to CSV - MODIFY THIS TO USE potential_fracture_endpoints
    export_endpoints_to_csv(points, potential_fracture_endpoints, component_labels, rib_numbers, output_csv)
    
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
    # print("\nComponent Analysis:")
    # for component in range(n_components):
    #    mean_coords = component_means[component]
    #    rib_id = rib_numbers.get(component, "Unassigned")
    #    print(f"Component {component} (Rib {rib_id}):")
    #    print(f"  Mean coordinates: X={mean_coords[0]:.2f}, Y={mean_coords[1]:.2f}, Z={mean_coords[2]:.2f}")
    
    # Print sorted rib grouping information
    # print("\nRib Groups:")
    # Create a dictionary to group components by rib ID
    rib_to_components = {}
    for group in rib_groups:
        if not group:  # Skip empty groups
            continue
        rib_id = rib_numbers.get(group[0], "Unassigned")
        rib_to_components[rib_id] = group
    
    # Create list of all possible rib IDs
    # all_rib_ids = ([f"R{i}" for i in range(1, 13)] + 
    #              [f"L{i}" for i in range(1, 13)])
    
    # Print all ribs in order
    # for rib_id in all_rib_ids:
    #    if rib_id in rib_to_components:
    #        group = rib_to_components[rib_id]
    #        print(f"Rib {rib_id}: Components {sorted(group)}")
    #        if len(group) > 1:
    #            print("  Potential fracture detected (multiple components)")
    #    else:
    #        print(f"Rib {rib_id}: No components (empty)")
    
    return n_components, combined_endpoints, rib_numbers, rib_groups

def export_endpoints_to_csv(points, combined_endpoints, component_labels, rib_numbers, output_csv):
    """
    Export the detected endpoints and their coordinates to a CSV file.
    
    Parameters:
    points (np.array): The point cloud data
    combined_endpoints (dict): Dictionary mapping component IDs to endpoint indices
    component_labels (np.array): Array indicating which component each point belongs to
    rib_numbers (dict): Dictionary mapping component IDs to rib designations
    output_csv (str): Path to output CSV file
    """
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['point_index', 'x', 'y', 'z', 'component_id', 'rib_designation', 'detection_method']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write data for each endpoint
        for component_id, endpoints_with_method in combined_endpoints.items():
            rib_designation = rib_numbers.get(component_id, "Unassigned")
            for endpoint_index, detection_method in endpoints_with_method:
                x, y, z = points[endpoint_index]
                writer.writerow({
                    'point_index': int(endpoint_index),
                    'x': float(x),
                    'y': float(y),
                    'z': float(z),
                    'component_id': int(component_id),
                    'rib_designation': rib_designation,
                    'detection_method': detection_method
                })
    
    print(f"Endpoint coordinates exported to {output_csv}")


def process_directory(directory_path, radius=None):
    """Process all OBJ and XYZ files in the given directory."""
    directory = Path(directory_path)
    results = {}
    
    output_dir = directory / "endpoint_filtered_analysis"
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
    
    # Create a master CSV with all endpoints from all files
    master_csv_path = output_dir / "all_endpoints.csv"
    with open(master_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'rib', 'component_id', 'fracture_detected', 'endpoint_count'])
        
        for filename, data in results.items():
            for comp, rib in data['rib_assignments'].items():
                # Find which group this component belongs to
                group = next((g for g in data['potential_fractures'] if comp in g), [])
                fracture_detected = len(group) > 1
                endpoint_count = data['endpoints_per_component'].get(comp, 0)
                
                writer.writerow([
                    filename,
                    rib,
                    comp,
                    'Yes' if fracture_detected else 'No',
                    endpoint_count
                ])
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")
    print(f"Endpoints summary saved to {master_csv_path}")


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