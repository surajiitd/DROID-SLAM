#!/bin/bash


ICL_PATH=../../Datasets/ICL_NUIM

evalset=(
    living_room/living_room_traj0_frei_png
    living_room/living_room_traj1_frei_png
    living_room/living_room_traj2_frei_png
    living_room/living_room_traj3_frei_png
    office_room/traj0_frei_png
    office_room/traj1_frei_png
    office_room/traj2_frei_png
    office_room/traj3_frei_png
)

for seq in ${evalset[@]}; do
    CUDA_VISIBLE_DEVICES=1 python evaluation_scripts/test_icl.py --stride=1 --modality=rgb --dataset_name=icl --datapath=$ICL_PATH/$seq --weights=droid.pth --disable_vis $@
done

#--modality = "rgb","rgbd","p-rgbd"
# also give --scaled_prgbd : for median scaling with p-rgbd

CUDA_VISIBLE_DEVICES=1 python evaluation_scripts/test_icl.py --stride=2 --modality=rgb --dataset_name=icl --datapath=../Datasets/ICL_NUIM/living_room/living_room_traj0_frei_png --weights=droid.pth --disable_vis