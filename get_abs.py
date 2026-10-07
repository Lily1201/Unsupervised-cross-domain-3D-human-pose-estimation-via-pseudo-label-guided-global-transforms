# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:27:28 2024

@author: 11052
"""


import numpy as np
import torch
from scipy.optimize import least_squares
def normalize_screen_coordinates(X, w, h): 
    assert X.shape[-1] == 2
    
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return (X+[1, h/w])/2*w
    # return X/w*2 - [1, h/w]
    
def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]
    
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
        
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]
    
    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2]**2, dim=len(XX.shape)-1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2**2, r2**3), dim=len(r2.shape)-1), dim=len(r2.shape)-1, keepdim=True)
    tan = torch.sum(p*XX, dim=len(XX.shape)-1, keepdim=True)

    XXX = XX*(radial + tan) + p*r2
    
    return f*XXX + c

pre=np.load('data/pre_3DHP.npy')
mpi3d_npz = np.load('data/test_3DHP_scaled.npz') 
tmp = mpi3d_npz
mpi3d= np.delete(tmp['pose_3d'], 9, axis=1)
mpi2d=np.delete(tmp['pose_2d'], 9, axis=1)
mpi2d[:2207,:, :2] = normalize_screen_coordinates(mpi2d[:2207,:, :2], w=2048, h=2048)
mpi2d[2207:,:, :2] = normalize_screen_coordinates(mpi2d[2207:,:, :2], w=1920, h=1080)
re_3d=mpi3d-mpi3d[:,:1,]

mpi3d_est=np.load('data/test_3DHP_est_fromest_5p.npz', allow_pickle=True)
est_3d=mpi3d_est['pose_3d']

cam1=torch.tensor([1500.172288  , 1500.172288  , 1017.38733568, 1043.03198208,
          0.        ,    0.        ,    0.        ,    0.        ,
          0.        ])
cam2=torch.tensor([1683.9834595, 1683.98345952, 939.85754016, 560.1407431,
              0.        ,    0.        ,    0.        ,    0.        ,
              0.        ])

def objective_function(depths, idd, relative_points, points_2d):
    residuals = []
    zp=depths+ relative_points
    if idd<2207:
        cam=cam1
    else:
        cam=cam2
    cam_params_source = cam.repeat(zp.shape[0], 1)
    proj_2d= project_to_2d(torch.tensor(zp),cam_params_source)
    
    for i in range(zp.shape[0]):
        residuals.append(
        np.linalg.norm(proj_2d[i] - points_2d[i]))  
    residuals= np.array(residuals)    
    return residuals


pre_3d=np.zeros(pre.shape)
err_3d=np.zeros(pre.shape)
for idd in range(re_3d.shape[0]):
    relative_points=pre[idd]
    points_2d=mpi2d[idd]
    gt_3d=mpi3d[idd]
    root=est_3d[idd,0,0]
    if root[2]<0:
        root*=-1
    print(idd)
    if idd<2207:
        result = least_squares(objective_function,  root, args=(idd,relative_points, points_2d))
    else:
        result = least_squares(objective_function, root, args=(idd, relative_points, points_2d))

    optimized_depths = result.x
    optimized_points=relative_points+optimized_depths
    err=(optimized_points-gt_3d)*1000
    pre_3d[idd]=optimized_points
    err_3d[idd]=err
    
e0=err_3d[:,0]   
np.save('data/pre_3DHP_abs.npy',pre_3d)    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
