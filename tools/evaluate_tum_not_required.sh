#!/bin/bash


TUM_PATH=../../Datasets/TUM_RGBD/extracted

evalset=(
    #rgbd_dataset_freiburg1_360
    rgbd_dataset_freiburg1_desk
    # rgbd_dataset_freiburg1_desk2
    # rgbd_dataset_freiburg1_floor
    # rgbd_dataset_freiburg1_plant
    # rgbd_dataset_freiburg1_room
    # rgbd_dataset_freiburg1_rpy
    # rgbd_dataset_freiburg1_teddy
    # rgbd_dataset_freiburg1_xyz
)

for seq in ${evalset[@]}; do
    python evaluation_scripts/test_tum.py --stride=2 --modality=rgb --dataset_name=tum --datapath=$TUM_PATH/$seq --weights=droid.pth --disable_vis $@
done


# --modality = "rgb", "rgbd" or "p-rgbd".
# --csv_suffix = extra suffix depending upon the experiment.

# allinone has for all above 3 modalities according to which flag you give.