import numpy as np

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
