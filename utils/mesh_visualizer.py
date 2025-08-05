import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
import trimesh
import io

def load_mesh(filepath):
    """Load mesh using trimesh first, then convert to Open3D"""
    try:
        print(f"Loading {filepath}...")
        
        # Use trimesh for robust loading
        trimesh_obj = trimesh.load(filepath)
        if isinstance(trimesh_obj, trimesh.Scene):
            trimesh_obj = list(trimesh_obj.geometry.values())[0]
        
        print(f"  Vertices: {len(trimesh_obj.vertices)}")
        if hasattr(trimesh_obj, 'faces'):
            print(f"  Faces: {len(trimesh_obj.faces)}")
        
        # Convert to Open3D
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(trimesh_obj.vertices)
        
        if hasattr(trimesh_obj, 'faces') and len(trimesh_obj.faces) > 0:
            mesh.triangles = o3d.utility.Vector3iVector(trimesh_obj.faces)
            print(f"  Loaded as mesh with {len(mesh.triangles)} triangles")
        else:
            print(f"  Loaded as point cloud")
        
        return mesh
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def render_mesh_o3d(mesh, resolution=(800, 800)):
    """Render mesh using Open3D's high-quality renderer"""
    
    # Check if mesh has any vertices
    if len(mesh.vertices) == 0:
        print("Warning: Empty mesh, creating blank image")
        return np.ones((resolution[1], resolution[0], 3), dtype=np.uint8) * 255
    
    # Check if it's a point cloud or mesh
    if len(mesh.triangles) == 0:
        # Convert to point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        geometry = pcd
        print("  Rendering as point cloud")
    else:
        # It's a mesh - apply nice material
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray
        geometry = mesh
        print("  Rendering as mesh")
    
    try:
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=resolution[0], height=resolution[1])
        
        # Add geometry
        vis.add_geometry(geometry)
        
        # Set up nice lighting and camera
        ctr = vis.get_view_control()
        
        # Get bounding box for camera positioning
        bbox = geometry.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()
        
        ctr.set_lookat(center)
        ctr.set_up([0, 0, 1])  # Z-up 
        ctr.set_front([0, -1, 0])  # Look from front (negative Y direction)
        
        # Set zoom
        ctr.set_zoom(0.7)
        
        # Render options for quality
        render_opt = vis.get_render_option()
        render_opt.mesh_show_back_face = True
        render_opt.light_on = True
        render_opt.point_size = 3.0
        
        # Update and capture
        vis.poll_events()
        vis.update_renderer()
        
        # Capture screen
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        
        # Convert to PIL Image
        image_np = np.asarray(image) * 255
        image_np = image_np.astype(np.uint8)
        
        return image_np
        
    except Exception as e:
        print(f"Open3D rendering failed: {e}")
        return np.ones((resolution[1], resolution[0], 3), dtype=np.uint8) * 255

def render_mesh_matplotlib_fallback(mesh_file):
    """Fallback to matplotlib if Open3D fails"""
    mesh = trimesh.load(mesh_file)
    if isinstance(mesh, trimesh.Scene):
        mesh = list(mesh.geometry.values())[0]
    
    fig = plt.figure(figsize=(8, 8), facecolor='white', dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    vertices = mesh.vertices
    bounds = mesh.bounds
    
    if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
        faces = mesh.faces
        face_vertices = vertices[faces]
        
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        collection = Poly3DCollection(face_vertices, 
                                    alpha=0.8, 
                                    facecolor='lightgray',
                                    edgecolor='gray',
                                    linewidth=0.1)
        ax.add_collection3d(collection)
    else:
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                  c='gray', s=1.0, alpha=0.8)
    
    ax.set_xlim(bounds[0, 0], bounds[1, 0])
    ax.set_ylim(bounds[0, 1], bounds[1, 1])
    ax.set_zlim(bounds[0, 2], bounds[1, 2])
    ax.set_box_aspect([1,1,1])
    ax.axis('off')
    ax.view_init(elev=0, azim=90)  # Front view, rotated 180 degrees
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return np.array(img)

def render_mesh(mesh_file, resolution=(800, 800)):
    """Main render function with fallback"""
    try:
        mesh = load_mesh(mesh_file)
        if mesh is not None:
            return render_mesh_o3d(mesh, resolution)
        else:
            print(f"Falling back to matplotlib for {mesh_file}")
            return render_mesh_matplotlib_fallback(mesh_file)
    except Exception as e:
        print(f"Open3D failed for {mesh_file}, using matplotlib: {e}")
        return render_mesh_matplotlib_fallback(mesh_file)

def create_figure(mesh_files, output_path, views_per_mesh=1, figsize=None):
    n_cols = len(mesh_files) * views_per_mesh
    
    if figsize is None:
        figsize = (4 * n_cols, 4)
    
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]
    
    col_idx = 0
    for mesh_file in mesh_files:
        for _ in range(views_per_mesh):
            img = render_mesh(mesh_file)
            axes[col_idx].imshow(img)
            axes[col_idx].axis('off')
            col_idx += 1
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved figure to: {output_path}")

def render_single(mesh_file, output_path):
    img = render_mesh(mesh_file, resolution=(1024, 1024))
    Image.fromarray(img).save(output_path, 'PNG')

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py output.png mesh1.ply mesh2.ply ...")
        sys.exit(1)
    
    output_path = sys.argv[1]
    mesh_files = sys.argv[2:]
    
    create_figure(mesh_files, output_path)

if __name__ == "__main__":
    main()