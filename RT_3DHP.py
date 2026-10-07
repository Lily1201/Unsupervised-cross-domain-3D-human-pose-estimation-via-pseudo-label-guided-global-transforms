# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 18:42:38 2024

@author: 11052
"""

import numpy as np


pose_3d=np.load('data/pre_3DHP_abs.npy')
valid=[]
for i in range(pose_3d.shape[0]):
    if np.sum(pose_3d[i])!=0:
        valid.append(i)

source_3d=pose_3d
j1_4=source_3d[:,1]-source_3d[:,4]
j0_8=source_3d[:,8]-(source_3d[:,1]+source_3d[:,4])/2
O_C_in_A = (source_3d[:,1]+source_3d[:,4])/2
i_C_in_A =j0_8/np.linalg.norm(j0_8, axis=1, keepdims=True)
pi=np.cross(j1_4,j0_8,axis=1)
pi=pi/np.linalg.norm(pi, axis=1, keepdims=True)
j_C_in_A =np.cross(i_C_in_A,pi, axis=1)
k_C_in_A = pi   
R = np.stack((i_C_in_A , j_C_in_A , k_C_in_A ),axis=2)
T = O_C_in_A
R_inv = np.linalg.inv(R)

for i in range(R.shape[0]):
    if not i in valid:
        R[i]=np.zeros((3,3))
        T[i]=np.zeros((3))
    
RT_PD={}
RT_PD.update({'R': R})
RT_PD.update({'T': T})
np.savez('data/RT_3DHP.npz',**RT_PD)