#!/bin/bash


# ETH_PATH=datasets/ETH3D-SLAM/training
ETH_PATH=../../Datasets/eth_3d/train

# all "non-dark" training scenes
# evalset=(
#     cables_1
#     cables_2
#     cables_3
#     camera_shake_1
#     camera_shake_2
#     camera_shake_3
#     ceiling_1
#     ceiling_2
#     desk_3
#     desk_changing_1
#     einstein_1
#     einstein_2
#     # einstein_dark
#     einstein_flashlight
#     einstein_global_light_changes_1
#     einstein_global_light_changes_2
#     einstein_global_light_changes_3
#     kidnap_1
#     # kidnap_dark
#     large_loop_1
#     mannequin_1
#     mannequin_3
#     mannequin_4
#     mannequin_5
#     mannequin_7
#     mannequin_face_1
#     mannequin_face_2
#     mannequin_face_3
#     mannequin_head
#     motion_1
#     planar_2
#     planar_3
#     plant_1
#     plant_2
#     plant_3
#     plant_4
#     plant_5
#     # plant_dark
#     plant_scene_1
#     plant_scene_2
#     plant_scene_3
#     reflective_1
#     repetitive
#     sfm_bench
#     sfm_garden
#     sfm_house_loop
#     sfm_lab_room_1
#     sfm_lab_room_2
#     sofa_1
#     sofa_2
#     sofa_3
#     sofa_4
#     # sofa_dark_1
#     # sofa_dark_2
#     # sofa_dark_3
#     sofa_shake
#     table_3
#     table_4
#     table_7
#     vicon_light_1
#     vicon_light_2
# )

# evalset=(
#     cables_1
#     cables_2
#     ceiling_1
#     desk_3
#     large_loop_1
#     mannequin_1
#     plant_scene_1
#     sfm_bench
#     sfm_garden
#     sfm_house_loop
#     sfm_lab_room_1
#     sofa_1
#     table_3
# )
evalset=(
    cables_1
    cables_2
    ceiling_1
    desk_3
    large_loop_1
)


for seq in ${evalset[@]}; do
    python evaluation_scripts/test_eth3d_allinone.py --csv_suffix=weight_one --dataset_name=eth3d --datapath=$ETH_PATH/$seq --weights=droid.pth --disable_vis $@
done


# Flags for different modalities
# for rgb    -> --mono
# for rgbd   -> (none)
# for p-rgbd -> --pred_depth

# So, I will use --csv_suffix flag as= "rgb", "rgbd", "p-rgbd", etc depending upon the experiment.

# allinone has for all above 3 modalities according to which flag you give.