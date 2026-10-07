# Unsupervised Cross-Domain 3D Human Pose Estimation via Pseudo-Label-Guided Global Transforms



This repository provides the implementation for unsupervised cross-domain 3D human pose estimation using pseudo-label-guided global transformations. The method leverages pseudo 3D poses from the target domain to estimate global transformations between camera-centric and human-centric coordinate systems, which are then used for cross-domain pose augmentation and adaptation.

## Repository Structure

```text
.
├── data/
│   ├── test_3DHP_scaled.npz
│   ├── data_3d_h36m.npz
│   ├── data_2d_h36m_gt.npz
│   └── pre_3DHP.npy
│
├── checkpoint/
│   └── ckpt_best_dhp_p1.pth.tar
│
├── function_baseline/
├── function_poseaug/
├── models_poseaug/
│
├── est_3dhp.py
├── get_abs.py
├── RT_3DHP.py
├── RT_H36M.py
├── run_poseaug.py
└── run_evaluate.py
```

## Installation

Create a Python environment and install the required dependencies.

```bash
conda create -n poseaug python=3.6.9
conda activate poseaug
pip install -r requirements.txt
```

The exact package versions used for the experiments will be provided in `requirements.txt`.

## Data Preparation

Create a `data/` directory and place the required Human3.6M and MPI-INF-3DHP data files inside it:

```text
data/
├── test_3DHP_scaled.npz
├── data_3d_h36m.npz
├── data_2d_h36m_gt.npz
└── pre_3DHP.npy
```

Here:

- `data_3d_h36m.npz` contains the Human3.6M 3D pose data.
- `data_2d_h36m_gt.npz` contains the corresponding Human3.6M 2D ground-truth poses.
- `test_3DHP_scaled.npz` contains the processed MPI-INF-3DHP data used for evaluation.
- `pre_3DHP.npy` contains the predicted 3D poses of MPI-INF-3DHP generated using a pre-trained pose estimator.

> **Note:** Dataset files are not distributed with this repository. Please follow PoseAug https://github.com/jfzhang95/PoseAug to prepare the Human3.6M dataset and 3DHP dataset.

### 1. Generate Target-Domain Pseudo 3D Poses

First, use the source-trained baseline model to generate 3D pose predictions for MPI-INF-3DHP.

Save the resulting predictions as:

```text
data/pre_3DHP.npy
```

These predictions are used as pseudo 3D poses for the subsequent target-domain processing.

### 2. Estimate Absolute 3D Poses

Estimate the absolute 3D poses of MPI-INF-3DHP using bone-length constraints:

```bash
python est_3dhp.py
python get_abs.py
```

### 3. Estimate Global Transformations for MPI-INF-3DHP

Estimate the rotation and translation matrices that transform MPI-INF-3DHP poses from the camera coordinate system to the human-centric coordinate system:

```bash
python RT_3DHP.py
```

### 4. Estimate Global Transformations for Human3.6M

Similarly, estimate the rotation and translation matrices for Human3.6M:

```bash
python RT_H36M.py
```

These transformations are subsequently used to transfer the global pose characteristics of the target domain to the source-domain poses during training.

## Pre-trained Model

Before running cross-domain adaptation, place the baseline model pre-trained on Human3.6M in the `checkpoint/` directory:

```text
checkpoint/
└── ckpt_best_dhp_p1.pth.tar
```

The pre-trained source model is used to initialise the pose estimator and to generate pseudo 3D poses for the target domain.

## Training

Run the cross-domain training using:

```bash
python3 run_poseaug.py \
    --note poseaug \
    --posenet_name videopose \
    --lr_p 1e-4 \
    --checkpoint ./checkpoint/RT1 \
    --keypoints gt
```

The trained models and intermediate checkpoints will be saved to the specified checkpoint directory.

## Evaluation

To evaluate a trained model on MPI-INF-3DHP, run:

```bash
python3 run_evaluate.py \
    --posenet_name videopose \
    --keypoints gt \
    --evaluate ./checkpoint/RT/ckpt_best_dhp_p1.pth.tar
```

Replace the checkpoint path with the model that you would like to evaluate.

## Method Pipeline

The main workflow can be summarised as:

```text
Human3.6M
   │
   ├── Source 2D/3D poses
   │
   └── Camera → Human-centric transformation
                     │
                     │
                     ▼
              Global Transformation
                     ▲
                     │
MPI-INF-3DHP         │
   │                 │
   ├── 2D poses      │
   │                 │
   └── Baseline 3D prediction
              │
              ▼
         Pseudo 3D Pose
              │
              ▼
      Absolute Pose Estimation
              │
              ▼
 Camera → Human-centric transformation
              │
              ▼
      Target-guided Augmentation
              │
              ▼
       Pose Estimator Training
```

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{liu2025unsupervised,
  title={Unsupervised Cross-Domain 3D Human Pose Estimation via Pseudo-Label-Guided Global Transforms},
  author={...},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025}
}
```

The complete BibTeX information will be updated with the final publication metadata.

## Acknowledgements

This implementation builds upon existing work in 3D human pose estimation and pose augmentation. We thank the authors of the relevant open-source projects and the Human3.6M and MPI-INF-3DHP datasets.

Detailed acknowledgements and links to the upstream repositories will be added before release.

## License

Please see `LICENSE` for details.
