import numpy as np
import nibabel as nib
import SimpleITK as sitk
import open3d as o3d
import os
from tqdm import tqdm
from skimage.measure import label, regionprops

# Script adapted from RibSeg's post_proc.py to create individual rib point clouds

source_data_dir = './data/ribfrac/ribfrac-test-images/'
dis_c2_point_dir = './inference_res/point/'
dis_c2_label_dir = './inference_res/label/'
output_ply_dir = './single_rib_pointclouds/'

# Create output directory if it doesn't exist
os.makedirs(output_ply_dir, exist_ok=True)

name_list = tqdm([x for x in os.listdir(source_data_dir)])

for ct in name_list:
    # Load source image
    s_i = nib.load(source_data_dir + ct)
    s_i = s_i.get_fdata()
    s_i[s_i != 0] = 1
    s_i = s_i.astype('int8')

    # Load point and label data
    loc = np.load(dis_c2_point_dir + ct[:-13]+'.npy')
    label = np.load(dis_c2_label_dir + ct[:-13]+'.npy')

    # Create masks
    mask_rd = np.zeros(s_i.shape)
    mask_res = np.zeros(s_i.shape)

    for index in loc:
        x, y, z = index[0], index[1], index[2]
        mask_rd[x][y][z] = 1
        mask_res[x][y][z] = label[loc.tolist().index(index)]

    # Image processing steps from original script
    lmage_array = sitk.GetImageFromArray(mask_res.astype('int8'))
    dilated = sitk.BinaryDilate(lmage_array, (3,3,3), sitk.sitkBall)
    holesfilled = sitk.GetArrayFromImage(dilated)

    res = np.multiply(s_i, holesfilled)
    res1 = label(res, connectivity=1)
    rib_p = regionprops(res1)
    rib_p.sort(key=lambda x: x.area, reverse=True)

    # Extract top 24 ribs
    im = np.in1d(res1, [x.label for x in rib_p[:24]]).reshape(res1.shape)
    im = im.astype('int8')

    # Extract and save individual rib point clouds
    for rib_num in range(1, 25):
        rib_mask = (im == rib_num)
        rib_points = np.argwhere(rib_mask)
        
        if rib_points.shape[0] > 0:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(rib_points)
            
            # Save as PLY with patient identifier
            filename = os.path.join(output_ply_dir, f'{ct[:-13]}_rib_{rib_num:02d}.ply')
            o3d.io.write_point_cloud(filename, pcd)