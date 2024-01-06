#!/bin/bash


ICL_PATH=../../Datasets/12scenes/tum_format

evalset=(
    apt1/kitchen/sequence0
    apt1/kitchen/sequence1
    apt1/kitchen/sequence2
    apt1/living/sequence0
    apt1/living/sequence1
    apt1/living/sequence2
    apt2/kitchen/sequence0
    apt2/kitchen/sequence1
    apt2/kitchen/sequence2
    apt2/kitchen/sequence3
    apt2/kitchen/sequence4
    apt2/bed/sequence0
    apt2/bed/sequence1
    apt2/bed/sequence2
    apt2/bed/sequence3
    apt2/bed/sequence4
    apt2/living/sequence0
    apt2/living/sequence1
    apt2/living/sequence2
    apt2/luke/sequence0
    apt2/luke/sequence1
    apt2/luke/sequence2
    apt2/luke/sequence3
    office1/gates362/sequence0
    office1/gates362/sequence1
    office1/gates362/sequence2
    office1/gates362/sequence3
    office1/gates362/sequence4
    office1/gates381/sequence0
    office1/gates381/sequence1
    office1/gates381/sequence2
    office1/gates381/sequence3
    office1/lounge/sequence0
    office1/lounge/sequence1
    office1/lounge/sequence2
    office1/lounge/sequence3
    office1/lounge/sequence4
    office1/manolis/sequence0
    office1/manolis/sequence1
    office1/manolis/sequence2
    office2/5a/sequence0
    office2/5a/sequence1
    office2/5a/sequence2
    office2/5b/sequence0
    office2/5b/sequence1
    office2/5b/sequence2
    office2/5b/sequence3
)

for seq in ${evalset[@]}; do
    CUDA_VISIBLE_DEVICES=6 python evaluation_scripts/test_12scenes.py --stride=1 --modality=rgb --dataset_name=12scenes --datapath=$ICL_PATH/$seq --weights=droid.pth --disable_vis $@
done
#--modality = "rgb","rgbd","p-rgbd"
# also give --scaled_prgbd : for median scaling with p-rgbd