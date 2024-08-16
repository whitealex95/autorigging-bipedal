# Auto-rigging 3D Bipedal Characters in Arbitrary Poses	
Pytorch code for the paper Auto-rigging 3D Bipedal Characters in Arbitrary Poses

**Jeonghwan Kim, Hyeontae Son, Jinseok Bae, Young Min Kim** 

## Prerequisites

#### Download Mixamo Dataset
- [Mixamo Dataset](https://drive.google.com/file/d/1d6o28Mu9yNaYCIWdZ-tiDDnTECnYDHzx/view?usp=sharing)


#### Install Dependencies
```
# Create conda environment
conda create -n autorigging python=3.7
conda activate autorigging

# Install torch-related
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-geometric

# Install other packages
pip install -r requirements.txt

```

## Usage
- Tested on Ubuntu 18.04

### Training
- You can train the model either by modifying the config files in `configs` directory or by using command-line arguments
    ```
    python train_vox.py -c configs/vox.yaml --log_dir=logs/vox
    python train_bvh.py -c configs/bvh.yaml --log_dir=logs/bvh
    python train_skin.py -c configs/skin.yaml --log_dir=logs/skin
    ```

- Settings used for our paper:
    ```
    python train_vox.py -c logs/vox.yaml --log_dir=logs/vox_ori_all_s4.5_HG_mean_stack2_down2_lr3e-4_b4_ce --loss_type=ce --batch_size=4 --downsample=2 --n_stack=2 --sigma=4.5 --mean_hourglass --data_dir=data/mixamo --no-reduce_motion --save_epoch=1

    python train_bvh.py -c configs/bvh.yaml --log_dir=logs/bvh/bvh_all_lr1e-3_zeroroot_bn --zero_root --bn --lr 1e-3 

    python train_skin.py -c configs/skin.yaml --workers=32 --batch_size=4 --lr=1e-4 --use_bn --log_dir=logs/skin/b4_tpl_and_euc0.12_lr1e-4_bn --euc_radius=0.12 --vis_step=10 --save_step=10 --edge_type tpl_and_euc
    ```

### Testing
    # Joint position prediction
    python test_vox.py -c logs/vox/vox_ori_all_s4.5_HG_mean_stack2_down2_lr3e-4_b4_ce/config.yaml --model=logs/vox/vox_ori_all_s4.5_HG_mean_stack2_down2_lr3e-4_b4_ce/model_epoch_030.pth

    # Joint rotation prediction
    python test_bvh.py --config logs/bvh/bvh_all_lr1e-3_zeroroot_bn/config.yaml --model logs/bvh/bvh_all_lr1e-3_zeroroot_bn/model_epoch_709.pth --joint_path logs/vox/vox_ori_all_s4.5_HG_mean_stack2_down2_lr3e-4_b4_ce/test

    # For calculating geodesic distance based on predicted joint position
    ## Note, geodesic distance for ground truth joint position is calculated in the dataset
    python prepare_vol_geo.py

    # Skin weight prediction
    python test_skin.py --config logs/skin/b4_tpl_and_euc0.12_lr1e-4_bn/config.yaml --model=logs/skin/b4_tpl_and_euc0.12_lr1e-4_bn/model_epoch_030.pth --vol_geo_dir logs/vox/vox_ori_all_s4.5_HG_mean_stack2_down2_lr3e-4_b4_ce/volumetric_geodesic_ours_final

### Evaluation
- Evaluation is done via `evaluate_rig.py`
- We have to put the result data in the right `dataset_dir` and add `--same_skeleton` when evaluating our model while omit the flag when evaluating rignet generated code.