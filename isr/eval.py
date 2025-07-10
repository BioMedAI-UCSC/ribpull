import open3d as o3d
import os
import utils
import numpy as np
import models
import torch
import opts
import pandas as pd
import numpy as np
import concurrent.futures
import time
from pathlib import Path

def load_inputs_with_bounds(shapepath, sigma, n_points):

    """
    Load input points and compute the bounding box of the pointcloud.

    Parameters
    ----------
    shapepath : str
        path to the shape data
    sigma : float
        level of noise to add to the pointcloud
    n_points : int
        number of points to sample from the pointcloud

    Returns
    -------
    shapedata : dict
        dictionary containing the pointcloud, occupancy grid, and pointcloud
    points_clean : (n_points, 3) array
        clean input points
    bounds : tuple of (3,) arrays
        bounding box of the pointcloud
    """
    shapedata = utils.load_pointcloud(shapepath)
    points_clean, _ = utils.sample_pointcloud(shapedata, N=n_points)
    noisy_points = utils.add_gaussian_noise(points_clean, sigma)
    bound_min = np.array([
        np.min(noisy_points[:, 0]), np.min(noisy_points[:, 1]),
        np.min(noisy_points[:, 2])
    ]) - 0.05
    bound_max = np.array([
        np.max(noisy_points[:, 0]), np.max(noisy_points[:, 1]),
        np.max(noisy_points[:, 2])
    ]) + 0.05
    return shapedata, points_clean, (bound_min, bound_max)


def load_occ_network( conf, ckpt, results_dir):
    occ_network = occ_network_from_conf( conf)
    occ_network.load_state_dict( torch.load (f'{results_dir}model_{ckpt}000.pth', map_location=torch.device("mps")) )
    return occ_network
def occ_network_from_conf( conf):
    occ_network = models.NPullNetwork(**conf['model.sdf_network'])
    bias = 0.5
    occ_network.lin8 = torch.nn.Linear(in_features=256, out_features=2, bias=True)
    return occ_network
def load_state_dict(  ckpt, results_dir):
    statedict = torch.load (f'{results_dir}model_{ckpt}000.pth', map_location=torch.device("mps"))
    return statedict

@torch.no_grad()
def uncertainty_inference (occ_network, pts):
    out = occ_network.sdf(pts.to(device)).softmax(1)
    return out[...,1]- out[...,0]

def select_ckpt(conf, args,input_points, bound_min, bound_max, ckpts):
    """
    Evaluate the given checkpoint numbers and return the one with the lowest chamfer distance wrt tot he input pointcloud.

    Parameters
    ----------
    conf : Config
        configuration object
    args : Namespace
        parsed command line arguments
    input_points : (n_points, 3) array
        input points
    bound_min : (3,) array
        lower bound of the bounding box
    bound_max : (3,) array
        upper bound of the bounding box
    ckpts : list of int
        list of checkpoint numbers to evaluate

    Returns
    -------
    best_ckpt : dict
        dictionary containing the best checkpoint number, chamfer distance, hausdorff distance, and the predicted mesh
    """
    def val_ckpt(ckpt):
        """
        Evaluate the given checkpoint numbers and return the one with the lowest chamfer distance wrt tot he input pointcloud.

        Parameters
        ----------
        conf : Config
            configuration object
        args : Namespace
            parsed command line arguments
        input_points : (n_points, 3) array
            input points
        bound_min : (3,) array
            lower bound of the bounding box
        bound_max : (3,) array
            upper bound of the bounding box
        ckpts : list of int
            list of checkpoint numbers to evaluate

        Returns
        -------
        best_ckpt : dict
            dictionary containing the best checkpoint number, chamfer distance, hausdorff distance, and the predicted mesh
        """
 
        occ_network = occ_network_from_conf( conf).to('mps')
        occ_network.load_state_dict( load_state_dict(  ckpt, args.results_dir) )
        inputs = torch.from_numpy(input_points).float()
        occ_function = lambda pts: uncertainty_inference (occ_network, pts)
        median_iso_level = occ_function (inputs ).median().detach().cpu().numpy()
        print("Median zero level set value: ", median_iso_level)
        cd1, hd, mesh,_= utils.validate_mesh(bound_min,bound_max, occ_function,  resolution=256, threshold=median_iso_level, 
                                            point_gt=input_points,
                                            N_val = 100000,
                                            compute_dist_fn=utils.compute_dists) 

        return {'cd1':cd1, 'hd':hd, 'mesh':mesh, 'network':occ_network}
    
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(val_ckpt, ckpt) for ckpt in ckpts]
        print('submit took: {:.2f} sec'.format(time.time() - start))
        start = time.time()
        scores = [future.result() for future in futures]
        print('result took: {:.2f} sec'.format(time.time() - start))
    best_idx = min(range(len(scores)), key=lambda i: scores[i]['cd1'])
    best_result = scores[best_idx]
    print(f"Best checkpoint: {ckpts[best_idx]}")
    return best_result

@torch.no_grad()
def save_slices(occ_network, bound_min, bound_max, heights, name, resolution=256):
    """Save cross-sectional slices at specified heights."""
    import matplotlib.pyplot as plt
    z_coords = np.linspace(bound_min[0], bound_max[0], resolution)
    y_coords = np.linspace(bound_min[1], bound_max[1], resolution)
    z_grid, y_grid = np.meshgrid(z_coords, y_coords, indexing='ij')
    
    for i, h in enumerate(heights):
        coords = np.stack([np.full_like(z_grid, h), y_grid, z_grid], axis=-1)
        coords_flat = torch.from_numpy(coords.reshape(-1, 3)).float().to('mps')
        out = occ_network.sdf(coords_flat).softmax(1)
        occupancy = (out[..., 1] - out[..., 0]).cpu().numpy().reshape(resolution, resolution)
        
        plt.figure(figsize=(8, 8))
        # Auto-scale to actual data range for better contrast
        vmin, vmax = occupancy.min(), occupancy.max()
        plt.imshow(occupancy.T, extent=[bound_min[0], bound_max[0], bound_min[1], bound_max[1]], 
                  origin='lower', cmap='RdBu', vmin=vmin, vmax=vmax)
        plt.colorbar(label=f'Occupancy [{vmin:.3f}, {vmax:.3f}]')
        plt.title(f'X-slice at height {h:.3f}')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.savefig(f'{name}_slice_{i}.jpg', dpi=150, bbox_inches='tight')
        plt.close()

def eval_pred_mesh(mesh, pointcloud_gt, normals_gt, n_points):
    """
    Evaluate a predicted mesh wrt a ground truth pointcloud.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        predicted mesh
    pointcloud_gt : (n_points, 3) array
        ground truth pointcloud
    normals_gt : (n_points, 3) array
        ground truth normals
    n_points : int
        number of points to sample from the mesh

    Returns
    -------
    scores : pandas.DataFrame
        a dataframe containing the chamfer-L1, chamfer-L2, and normals scores
    """
    pred_mesh_o3d = o3d.geometry.TriangleMesh( o3d.utility.Vector3dVector( mesh.vertices),
                              o3d.utility.Vector3iVector( mesh.faces) )
    pointcloud_pred = pred_mesh_o3d.sample_points_uniformly(n_points,use_triangle_normal=True)
    normals_pred = np.array(pointcloud_pred.normals).astype(np.float32)
    pointcloud_pred =np.array( pointcloud_pred.points).astype(np.float32)
    out_dict = utils.eval_pointcloud(pointcloud_pred, pointcloud_gt, normals_pred, normals_gt)
    return pd.DataFrame(out_dict, index = ["1"])[['chamfer-L1','chamfer-L2', 'normals']]

if __name__ == '__main__':  
    parser = opts.neural_pull_opts()
    parser.add_argument('--results_dir','-r', type=str, default='npull')
    parser.add_argument('--shapename', '-s',type=str, default='copyroom')
    parser.add_argument('--slice_heights', nargs='+', type=float, default=None, help='Heights for cross-sectional slices')
    args = parser.parse_args()
    #args.device
    os.environ['CUDA_VISIBLE_DEVICES']= str(args.device)
    conf = utils.load_conf(args.config)
    utils.fix_seeds()
    shapepath = args.shapename
    device = 'mps'
    shapedata ,input_points, (bound_min, bound_max) = load_inputs_with_bounds(shapepath, n_points = 1024, sigma = 0.0)

    #occ_network = load_occ_network( conf, ckpt = 35, results_dir = args.results_dir).to(device)
    ckpts = range(20, 41, 5)
    best_score = select_ckpt(conf, args, input_points, bound_min, bound_max, ckpts)
    
    # Save the best mesh
    name = Path(shapepath).name.split('-')[0]  # "RibFrac2"
    best_score['mesh'].export(name + '_best_mesh.obj')
    print(f"Best mesh saved as best_mesh.obj")

    # Generate slices using best checkpoint network
    heights = args.slice_heights if args.slice_heights else np.linspace(bound_min[0], bound_max[0], 5)
    save_slices(best_score['network'], bound_min, bound_max, heights, name)
    print(f"Slices saved as {name}_slice_*.jpg")
    
    print(eval_pred_mesh(best_score['mesh'], shapedata['pc'], shapedata['normals'], conf.get_int('val.n_val')))