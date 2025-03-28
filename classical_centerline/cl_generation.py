import os
import numpy as np
import SimpleITK as sitk
import skimage
import nibabel as nib
import kimimaro
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline
from skel import *

def skel_extraction(vol):
    skels = kimimaro.skeletonize(
    vol.astype(np.int16), 
    teasar_params={
        'scale': 4,
        'const': 500, # physical units
        'pdrf_exponent': 4,
        'pdrf_scale': 100000,
        'soma_detection_threshold': 1100, # physical units
        'soma_acceptance_threshold': 3500, # physical units
        'soma_invalidation_scale': 1.0,
        'soma_invalidation_const': 300, # physical units
        'max_paths': 50, # default None
    },
    # object_ids=[25024949], # process only the specified labels
    # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
    # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
    dust_threshold=1000, # skip connected components with fewer than this many voxels
    anisotropy=(1,1,1), # default True
    fix_branching=True, # default True
    fix_borders=True, # default True
    fill_holes=False, # default False
    fix_avocados=False, # default False
    progress=False, # default False, show progress bar
    parallel=1, # <= 0 all cpu, 1 single process, 2+ multiprocess
    parallel_chunk_size=10, # how many skeletons to process before updating progress bar
    )
    return  skels

def smooth_3d_array(points,num=None,**kwargs):
    x,y,z = points[:,0],points[:,1],points[:,2]
    points = np.zeros((num,3))
    if num is None:
        num = len(x)
    w = np.arange(0,len(x),1)
    sx = UnivariateSpline(w,x,**kwargs)
    sy = UnivariateSpline(w,y,**kwargs)
    sz = UnivariateSpline(w,z,**kwargs)
    wnew = np.linspace(0,len(x),num)
    points[:,0] = sx(wnew)
    points[:,1] = sy(wnew)
    points[:,2] = sz(wnew)
    return points

def dilation(vol_ct,vol_rib):
    '''
    input:
        vol_ct: raw ct volume
        vol_rib: predicted rib volume
    output:
        res:  delated rib volume's largest component
    '''
    img_array = sitk.GetImageFromArray(vol_rib.astype(np.int8))
    img_dilated = sitk.BinaryDilate(img_array,(5,5,5))
    mask_dilated = sitk.GetArrayFromImage(img_dilated)
    vol_rib_dilated = np.multiply(vol_ct,mask_dilated).astype(np.int8)

    vol_tmp = skimage.measure.label(vol_rib_dilated,connectivity=1)
    vol_region = skimage.measure.regionprops(vol_tmp)
    vol_region.sort(key=lambda x: x.area, reverse = True)
    res = np.in1d(vol_tmp,[x.label for x in vol_region[:1]]).reshape(vol_rib_dilated.shape)
    return res


dir_pred = '../../RibSeg/inference_res_compressed/'
dir_ct = '../../RibSeg/data/pn/data_pn/test/'
dir_save = './cl_upsample/'
if not os.path.exists(dir_save):
    os.makedirs(dir_save)

list_pred = [x for x in os.listdir(dir_pred)]
for name in tqdm(list_pred):
    data = np.load(dir_pred + name)
    pc_ct, label_pred = data['ct'].astype(np.int16),data['seg'].astype(np.int8)

    vol_ct = nib.load(dir_ct+name[:-4]+'-rib-binary.nii.gz').get_fdata()

    vol_pred = np.zeros(vol_ct.shape).astype(np.int8)
    for idx in range(pc_ct.shape[0]):
        cords = pc_ct[idx]
        x,y,z = cords[0],cords[1],cords[2]
        vol_pred[x][y][z] = label_pred[idx]

    cl_data = {}
    cl_org = {}
    cl = np.zeros((24,500,3))
    for i in range(1,25):
        vol_tmp = vol_pred.copy()
        vol_tmp[vol_tmp!=i]=0
        vol_tmp[vol_tmp!=0]=1

        vol_rib = dilation(vol_ct,vol_tmp).astype(np.int8)
        try:
            tmp_cl = skel_extraction(vol_rib)[1]
            cl_data[i] = tmp_cl
            seed = find_furthest_pt(tmp_cl, 0, single=False)[0]
            longest_path = find_furthest_pt(tmp_cl, seed, single=False)[1][0]
            org = tmp_cl.vertices[longest_path]
            aug = smooth_3d_array(org,num = 500,s=2000)
            cl[i-1] = aug
            cl_org[i] = org
        except:
            cl[i-1,0,0]=-1
            print(name,"has no",i)
    np.savez_compressed(dir_save + name[:-4], cl=cl,cl_org=cl_org,cl_data = cl_data,size_vol = vol_ct.shape)


