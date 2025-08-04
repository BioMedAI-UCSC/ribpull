import sys
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import io

def load_mesh(filepath):
    mesh = trimesh.load(filepath)
    if isinstance(mesh, trimesh.Scene):
        mesh = list(mesh.geometry.values())[0]
    return mesh

def render_mesh(mesh, resolution=(800, 800)):
    # Use matplotlib 3D rendering (no OpenGL dependencies)
    fig = plt.figure(figsize=(8, 8), facecolor='white', dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    vertices = mesh.vertices
    bounds = mesh.bounds
    
    # Check if it's a mesh with faces or just a point cloud
    if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
        # Render as mesh
        faces = mesh.faces
        face_vertices = vertices[faces]
        collection = Poly3DCollection(face_vertices, alpha=0.7, facecolor='lightgray', 
                                     edgecolor='gray', linewidth=0.1)
        ax.add_collection3d(collection)
    else:
        # Render as point cloud
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                  c='gray', s=0.5, alpha=0.8)
    
    # Set equal aspect ratio and limits
    ax.set_xlim(bounds[0, 0], bounds[1, 0])
    ax.set_ylim(bounds[0, 1], bounds[1, 1])
    ax.set_zlim(bounds[0, 2], bounds[1, 2])
    ax.set_box_aspect([1,1,1])
    
    # Clean appearance
    ax.axis('off')
    ax.grid(False)
    
    # Set viewing angle for proper ribcage orientation (vertical, chest-like view)
    ax.view_init(elev=0, azim=-90)  # Front view, ribs should be vertical
    
    # Save to PIL Image instead of buffer conversion
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return np.array(img)

def create_figure(mesh_files, output_path, views_per_mesh=1, figsize=None):
    meshes = [load_mesh(f) for f in mesh_files]
    n_cols = len(meshes) * views_per_mesh
    
    if figsize is None:
        figsize = (4 * n_cols, 4)
    
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]
    
    col_idx = 0
    for mesh in meshes:
        for _ in range(views_per_mesh):
            img = render_mesh(mesh)
            axes[col_idx].imshow(img)
            axes[col_idx].axis('off')
            col_idx += 1
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved figure to: {output_path}")

def render_single(mesh_file, output_path):
    mesh = load_mesh(mesh_file)
    img = render_mesh(mesh, resolution=(1024, 1024))
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