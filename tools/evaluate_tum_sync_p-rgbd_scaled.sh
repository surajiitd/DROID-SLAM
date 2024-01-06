#!/bin/bash


TUM_PATH=../../Datasets/TUM_RGBD/extracted

evalset=(
    #test:
    rgbd_dataset_freiburg1_360
    rgbd_dataset_freiburg1_desk
    rgbd_dataset_freiburg1_desk2
    rgbd_dataset_freiburg1_floor
    rgbd_dataset_freiburg1_plant
    rgbd_dataset_freiburg1_room
    rgbd_dataset_freiburg1_rpy
    rgbd_dataset_freiburg1_teddy
    rgbd_dataset_freiburg1_xyz
    #train:
    rgbd_dataset_freiburg2_360_hemisphere
    rgbd_dataset_freiburg2_360_kidnap
    rgbd_dataset_freiburg2_coke
    rgbd_dataset_freiburg2_dishes
    rgbd_dataset_freiburg2_flowerbouquet
    rgbd_dataset_freiburg2_large_no_loop
    rgbd_dataset_freiburg3_sitting_rpy
    rgbd_dataset_freiburg3_long_office_household
)

for seq in ${evalset[@]}; do
    python evaluation_scripts/test_tum.py --scaled_prgbd --sync --stride=2 --modality=p-rgbd --dataset_name=tum --datapath=$TUM_PATH/$seq --weights=droid.pth --disable_vis $@
done

#--modality = "rgb","rgbd","p-rgbd"
# also give --scaled_prgbd : for median scaling with p-rgbd
# --sync is used only for TUM