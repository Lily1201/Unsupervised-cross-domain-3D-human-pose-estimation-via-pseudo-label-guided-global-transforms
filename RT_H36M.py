# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:31:35 2024

@author: 11052
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from common.camera import world_to_camera
from common.h36m_dataset import Human36mDataset

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


dataset_path = 'data/data_3d_h36m.npz'

dataset = Human36mDataset(dataset_path)
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        positions_3d = []
        for cam in anim['cameras']:
            pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
            # pos_3d[:, :] -= pos_3d[:, :1]  # keep this, remove at model training.
            positions_3d.append(pos_3d)
        anim['positions_3d'] = positions_3d


RT_H36M={}

for sub_key in dataset.subjects():
    subject=sub_key
    RT_actions={}
    for action in dataset[subject].keys():
      anim = dataset[subject][action]
      act_3d=anim['positions_3d']
      R_cams=[]
      T_cams=[]
      for i in range(len(act_3d)):  
        source_3d=act_3d[i]

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
        R_cams.append(R_inv)
        T_cams.append(T)
      RT_actions.update({action+'_Rinv': R_cams})
      RT_actions.update({action+'_T': T_cams})
        # Pnew=np.zeros(source_3d.shape)
        # for i in range(17):
        #     P_A=source_3d[:,i]
        #     P_C = np.einsum('ijk,ik->ij', R_inv, P_A - T)
        #     Pnew[:,i]=P_C
    RT_H36M.update({sub_key: RT_actions})
np.savez('data/RT_H36M.npz',**RT_H36M )