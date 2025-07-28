import numpy as np
from scipy.spatial import KDTree
import networkx as nx

class L1SkeletonExtractor:
    def __init__(self, distance_threshold=2.0, connectivity_radius=3.0, min_branch_length=5):
        self.distance_threshold = distance_threshold
        self.connectivity_radius = connectivity_radius
        self.min_branch_length = min_branch_length
        
    def l1_distance(self, p1, p2):
        """L1 (Manhattan) distance"""
        return np.sum(np.abs(p1 - p2), axis=-1)
    
    def find_boundary_points(self, points, k=20):
        """Find boundary points using local density"""
        tree = KDTree(points)
        distances, _ = tree.query(points, k=k+1)
        local_density = np.mean(distances[:, 1:], axis=1)
        threshold = np.percentile(local_density, 75)
        return points[local_density > threshold]
    
    def compute_distance_field(self, points, boundary_points):
        """Distance from each point to nearest boundary"""
        distances = np.zeros(len(points))
        for i, point in enumerate(points):
            dists = self.l1_distance(boundary_points, point)
            distances[i] = np.min(dists)
        return distances
    
    def find_medial_points(self, points, distance_field, radius=2.0):
        """Find local maxima (medial points)"""
        tree = KDTree(points)
        medial_indices = []
        
        for i, point in enumerate(points):
            neighbors = tree.query_ball_point(point, radius)
            neighbor_dists = distance_field[neighbors]
            
            if (distance_field[i] >= np.max(neighbor_dists) and 
                distance_field[i] > self.distance_threshold):
                medial_indices.append(i)
                
        return np.array(medial_indices)
    
    def build_skeleton_graph(self, medial_points):
        """Connect nearby medial points"""
        tree = KDTree(medial_points)
        G = nx.Graph()
        
        for i, point in enumerate(medial_points):
            G.add_node(i, pos=point)
        
        for i, point in enumerate(medial_points):
            neighbors = tree.query_ball_point(point, self.connectivity_radius)
            for j in neighbors:
                if i != j:
                    G.add_edge(i, j)
        return G
    
    def prune_skeleton(self, G):
        """Remove small components and short branches"""
        # Keep only large components
        components = list(nx.connected_components(G))
        large_comps = [c for c in components if len(c) >= self.min_branch_length]
        nodes_to_keep = set().union(*large_comps) if large_comps else set()
        
        return G.subgraph(nodes_to_keep).copy()
    
    def extract_skeleton(self, points):
        """Main extraction pipeline"""
        print(f"Processing {len(points)} points...")
        
        boundary_points = self.find_boundary_points(points)
        print(f"Found {len(boundary_points)} boundary points")
        
        distance_field = self.compute_distance_field(points, boundary_points)
        
        medial_indices = self.find_medial_points(points, distance_field)
        medial_points = points[medial_indices]
        print(f"Found {len(medial_points)} medial points")
        
        if len(medial_points) < 2:
            return None, None
        
        skeleton_graph = self.build_skeleton_graph(medial_points)
        skeleton_graph = self.prune_skeleton(skeleton_graph)
        
        print(f"Final skeleton: {skeleton_graph.number_of_nodes()} nodes, {skeleton_graph.number_of_edges()} edges")
        return skeleton_graph, medial_points

def load_ply(filename):
    """Load PLY point cloud"""
    points = []
    with open(filename, 'rb') as f:
        # Read header
        line = f.readline().decode('utf-8').strip()
        if line != 'ply':
            raise ValueError("Not a PLY file")
        
        vertex_count = 0
        properties = []
        format_type = None
        
        while True:
            line = f.readline().decode('utf-8').strip()
            if line.startswith('format'):
                format_type = line.split()[1]
            elif line.startswith('element vertex'):
                vertex_count = int(line.split()[2])
            elif line.startswith('property'):
                properties.append(line.split()[2])
            elif line == 'end_header':
                break
        
        # Find coordinate indices
        x_idx, y_idx, z_idx = properties.index('x'), properties.index('y'), properties.index('z')
        
        # Read vertices
        if format_type == 'ascii':
            for _ in range(vertex_count):
                line = f.readline().decode('utf-8').strip()
                values = line.split()
                points.append([float(values[x_idx]), float(values[y_idx]), float(values[z_idx])])
        else:  # binary
            import struct
            for _ in range(vertex_count):
                vertex_data = []
                for _ in properties:
                    vertex_data.append(struct.unpack('<f', f.read(4))[0])
                points.append([vertex_data[x_idx], vertex_data[y_idx], vertex_data[z_idx]])
    
    return np.array(points)

def save_ply(skeleton_graph, filename):
    """Save skeleton as PLY file"""
    # Extract points and edges
    points = np.array([skeleton_graph.nodes[node]['pos'] for node in skeleton_graph.nodes()])
    node_map = {node: i for i, node in enumerate(skeleton_graph.nodes())}
    edges = [[node_map[e[0]], node_map[e[1]]] for e in skeleton_graph.edges()]
    
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")
        
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        for edge in edges:
            f.write(f"{edge[0]} {edge[1]}\n")
    
    print(f"Skeleton saved to {filename}")
    return points, edges

def visualize(points, skeleton_graph, medial_points=None):
    """Visualize results"""
    fig = plt.figure(figsize=(15, 5))
    
    # Original points
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='gray', alpha=0.3, s=1)
    ax1.set_title('Original Point Cloud')
    
    # Medial points
    if medial_points is not None:
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c='gray', alpha=0.1, s=1)
        ax2.scatter(medial_points[:, 0], medial_points[:, 1], medial_points[:, 2], c='red', s=10)
        ax2.set_title('Medial Points')
    
    # Skeleton
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(points[:, 0], points[:, 1], points[:, 2], c='gray', alpha=0.1, s=1)
    
    # Draw skeleton
    skel_points = np.array([skeleton_graph.nodes[n]['pos'] for n in skeleton_graph.nodes()])
    ax3.scatter(skel_points[:, 0], skel_points[:, 1], skel_points[:, 2], c='blue', s=20)
    
    for edge in skeleton_graph.edges():
        p1, p2 = skeleton_graph.nodes[edge[0]]['pos'], skeleton_graph.nodes[edge[1]]['pos']
        ax3.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'r-', linewidth=2)
    
    ax3.set_title('L1-Medial Skeleton')
    
    plt.tight_layout()
    plt.savefig('skeleton_result.png', dpi=150)
    plt.show()

def main():
    # Configuration
    input_file = "RibFrac1-rib-seg_mesh.ply"
    output_file = "skeleton_output.ply"
    
    try:
        # Load and process
        points = load_ply(input_file)
        print(f"Loaded {len(points)} points")
        
        extractor = L1SkeletonExtractor(
            distance_threshold=1.5,
            connectivity_radius=3.0,
            min_branch_length=5
        )
        
        skeleton_graph, medial_points = extractor.extract_skeleton(points)
        
        if skeleton_graph is not None:
            skel_points, edges = save_ply(skeleton_graph, output_file)
            visualize(points, skeleton_graph, medial_points)
            
            print(f"\nResults:")
            print(f"  Original: {len(points)} points")
            print(f"  Skeleton: {len(skel_points)} points, {len(edges)} edges")
            print(f"  Compression: {len(skel_points)/len(points):.3f}")
        else:
            print("Failed to extract skeleton. Try adjusting parameters.")
            
    except FileNotFoundError:
        print(f"File {input_file} not found.")

if __name__ == "__main__":
    main()