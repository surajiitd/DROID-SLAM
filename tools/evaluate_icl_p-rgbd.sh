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
    CUDA_VISIBLE_DEVICES=6 python evaluation_scripts/test_icl.py --stride=1 --modality=p-rgbd --dataset_name=icl --datapath=$ICL_PATH/$seq --weights=droid.pth --disable_vis $@
done

#--modality = "rgb","rgbd","p-rgbd"
# but for p-rgbd_scaled : you have to pass --scaled_prgbd