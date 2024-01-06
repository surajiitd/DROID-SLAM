#!/bin/bash


EUROC_PATH=../../Datasets/euroc_mav/ASL

evalset=(
    MH_01_easy
    MH_02_easy
    MH_03_medium
    MH_04_difficult
    MH_05_difficult
    V1_01_easy
    V1_02_medium
    V1_03_difficult
    V2_01_easy
    V2_02_medium
    V2_03_difficult
)

for seq in ${evalset[@]}; do
    python evaluation_scripts/test_euroc.py --datapath=$EUROC_PATH/$seq --gt=data/euroc_groundtruth/$seq.txt --weights=../../DROID-SLAM/checkpoints/reproduce_250000.pth --disable_vis $@
done

