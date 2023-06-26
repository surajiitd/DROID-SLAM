#!/bin/bash

ROOT_PATH=/home/suraj/scratch/Datasets/7scenes
evalset=(
    office/seq-05
    office/seq-10
    redkitchen/seq-11
    pumpkin/seq-06
    heads/seq-02
    # #cables_2
    # #ceiling_1
    # #desk_3
    # #mannequin_1
    # sofa_1
    #sofa_1
    #sofa_1_long_fbfb
    #table_3
    #table_3_long
    #table_3_long_fbfb
    # sfm_house_loop
    # plant_scene_1

    #sfm_lab_room_1
    #large_loop_1
    #sfm_garden
    #sfm_bench

)


for seq in ${evalset[@]}; do
    echo cmd is: python evaluation_scripts/test_7scenes.py --datapath=$ROOT_PATH/$seq --weights=droid.pth --disable_vis $@
    CUDA_VISIBLE_DEVICES=4 python evaluation_scripts/test_7scenes.py --datapath=$ROOT_PATH/$seq --weights=droid.pth --disable_vis $@
done
# the last $@ is for the extra command-line-params that we've given got added automatically with that python command.



