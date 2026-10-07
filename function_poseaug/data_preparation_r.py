from __future__ import print_function, absolute_import, division

import os.path as path
import copy
import numpy as np
from torch.utils.data import DataLoader

from common.data_loader_r import PoseDataSet, PoseBuffer, PoseTarget, PoseDataSet_ori
from utils.data_utils import fetch, read_3d_data, create_2d_data

'''
this code is used for prepare data loader
'''


def data_preparation(args):
    """
    load the h36m dataset
    generate data loader for training posenet, poseaug, and cross-data evaluation
    """
    dataset_path = path.join('data', 'data_3d_' + args.dataset + '.npz')
    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset, TEST_SUBJECTS
        dataset = Human36mDataset(dataset_path)
        if args.s1only:
            subjects_train = ['S1']
        else:
            subjects_train = ['S1', 'S5', 'S6', 'S7', 'S8']
        subjects_test = TEST_SUBJECTS
    else:
        raise KeyError('Invalid dataset')

    print('==> Loading 3D data...')
    dataset = read_3d_data(dataset)

    print('==> Loading 2D detections...')
    keypoints = create_2d_data(path.join('data', 'data_2d_' + args.dataset + '_' + args.keypoints + '.npz'), dataset)

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        action_filter = map(lambda x: dataset.define_actions(x)[0], action_filter)
        print('==> Selected actions: {}'.format(action_filter))

    stride = args.downsample

    ############################################
    # general 2D-3D pair dataset
    ############################################
    RT=np.load('data/RT_H36M.npz',allow_pickle=True)
    RT_H36M={}
    for key in RT.keys():
        b=RT[key][()]
        ##exclude nose/neck
        RT_H36M.update({key: b})
    
    poses_train, poses_train_2d, actions_train, cams_train, R_train ,T_train = fetch(subjects_train, dataset,RT_H36M, keypoints, action_filter,
                                                                   stride)
    poses_valid, poses_valid_2d, actions_valid, cams_valid, R_valid, T_valid = fetch(subjects_test, dataset,RT_H36M, keypoints, action_filter,
                                                                   stride)
    # prepare train loader for detected 2D.
    train_det2d3d_loader = DataLoader(PoseDataSet_ori(poses_train, poses_train_2d,actions_train, cams_train),
                                      batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # prepare train loader for GT 2D - 3D, which will update by using projection.
    train_gt2d3d_loader = DataLoader(PoseDataSet(poses_train, poses_train_2d,R_train ,T_train, actions_train, cams_train),
                                     batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers, pin_memory=True)

    valid_loader = DataLoader(PoseDataSet(poses_valid, poses_valid_2d, R_valid ,T_valid,actions_valid, cams_valid),
                              batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    ############################################
    # data loader for GAN training
    ############################################
    target_2d_loader = DataLoader(PoseTarget(poses_train_2d),
                                  batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, pin_memory=True)
    target_3d_loader = DataLoader(PoseTarget(poses_train),
                                  batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, pin_memory=True)

    ############################################
    # prepare cross dataset validation
    ############################################
    # 3DHP -  2929 version
    mpi3d_npz = np.load('data/test_3DHP_scaled.npz')    # this is the 2929 version
    tmp = mpi3d_npz
    ##exclude nose/neck
    tmp_pose3d= np.delete(tmp['pose_3d'], 9, axis=1)
    tmp_pose2d= np.delete(tmp['pose_2d'], 9, axis=1)
    tmp_pose3d-=tmp_pose3d[:,:1,:]
    mpi3d_a=np.load('data/RT_3DHP.npz',allow_pickle=True)
    Rs=mpi3d_a['R']
    Ts=mpi3d_a['T']
    mpi3d_loader = DataLoader(PoseBuffer([tmp_pose3d], [tmp_pose2d],[Rs], [Ts] ),
                              batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return {
        'dataset': dataset,
        'train_det2d3d_loader': train_det2d3d_loader,
        'train_gt2d3d_loader': train_gt2d3d_loader,
        'target_2d_loader': target_2d_loader,
        'target_3d_loader': target_3d_loader,
        'H36M_test': valid_loader,
        'mpi3d_loader': mpi3d_loader,
        'action_filter': action_filter,
        'subjects_test': subjects_test,
        'keypoints': keypoints,
    }
