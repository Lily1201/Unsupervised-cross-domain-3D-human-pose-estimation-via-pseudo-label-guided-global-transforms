# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 23:30:58 2024

@author: 11052
"""

### generate estimated 3dhp dataset using gt skeleton length and camera intrinsics
import numpy as np
import torch
from scipy.optimize import least_squares
def normalize_screen_coordinates(X, w, h): 
    assert X.shape[-1] == 2
    
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return (X+[1, h/w])/2*w
    # return X/w*2 - [1, h/w]
def normalize_image_points(K, image_points):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    normalized_points = []
    for u, v in image_points:
        normalized_points.append([(u - cx) / fx, (v - cy) / fy] )
    return np.array(normalized_points)
def objective_function(depths, normalized_points, distances):
    residuals = []
    num_points = normalized_points.shape[0]
    zp=[]
    for i in range(num_points):
        Pi = depths[i] * np.append(normalized_points[i], 1)
        zp.append(Pi)
    estimated_distance = []
    for a,b in lines:
        estimated_distance.append(
        np.linalg.norm(zp[a] - zp[b]))
    residuals= np.array(estimated_distance)- np.array(distances)
    
    return residuals

def get_3d_points(normalized_points, depths):
    return np.array([depth * np.append(normalized_point, 1) for normalized_point, depth in zip(normalized_points, depths)])

lines = [
    (9, 10),  # 线1：点1到点2
    (8, 9),  # 线2：点2到点3
    (7, 8),  # 线3：点3到点4
    (7, 0),  # 线4：点4到点1
    (0, 1),  # 线5：点1到点5
    (0, 4),  # 线6：点2到点5
    (2, 1),  # 线7：点3到点5
    (3, 2),   # 线8：点4到点5
    (4,5),
    (5,6),
    (8,14),
    (8,11),
    (14,15),
    (15,16),
    (11,12),
    (12,13),
    (0,8),
    (1,4),
]

K1 = np.array([[1500.172288, 0, 1017.387],
              [0, 1500.172288, 1043.032],
              [0, 0, 1]])
K2 = np.array([[1683.9834595, 0, 939.85754016],
              [0, 1683.98345952, 560.1407431],
              [0, 0, 1]])
mpi3d_npz = np.load('data/test_3DHP_scaled.npz')    # this is the 2929 version
tmp = mpi3d_npz

mpi3d=tmp['pose_3d']
mpi2d=tmp['pose_2d']
mpi2d_ori=tmp['pose_2d']
mpi3d_est=[]
mpi2d[:2207,:, :2] = normalize_screen_coordinates(mpi2d[:2207,:, :2], w=2048, h=2048)
mpi2d[2207:,:, :2] = normalize_screen_coordinates(mpi2d[2207:,:, :2], w=1920, h=1080)
jind=[i for i in range(17)]


dist=np.zeros((mpi3d.shape[0],len(lines)))
for k in range(dist.shape[0]):
    for i in range(len(lines)):
        a=lines[i][0]
        b=lines[i][1]
        dist[k,i]=np.linalg.norm(mpi3d[k,a]-mpi3d[k,b])
    
dist_std=np.std(dist,axis=0)
dist_mean=np.mean(dist,axis=0)
dist=dist_mean

init_dep=[1,2,3,4,5]
for i in range(mpi3d.shape[0]):   

    image_points = mpi2d[i,jind]
    if i<2207:
        normalized_points =  normalize_image_points(K1, image_points)
    else:
        normalized_points =  normalize_image_points(K2, image_points)

    all_est=[]
    for d in init_dep:
        initial_depths = np.ones(len(image_points))*d
        result = least_squares(objective_function, initial_depths, args=(normalized_points, dist))

        optimized_depths = result.x
        points_3d = get_3d_points(normalized_points, optimized_depths)
        all_est.append(points_3d)
    mpi3d_est.append(all_est)
    print(i)
mpi3d_tmp=np.array(mpi3d_est)
valid={}
valid.update({'pose_2d':mpi2d_ori})
valid.update({'pose_3d':mpi3d_tmp})
np.savez('data/test_3DHP_est_fromest_5p.npz',**valid) 

