import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

import torch.nn.functional as F
from droid import Droid

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(datapath, use_depth=False, stride=1, pred_depth=False):
    """ image generator """

    fx, fy, cx, cy = np.loadtxt(os.path.join(datapath, 'calibration.txt')).tolist()
    image_list = sorted(glob.glob(os.path.join(datapath, 'rgb', '*.png')))[::stride]
    if pred_depth:
        depth_list = sorted(glob.glob(os.path.join(datapath, 'pixelformer_depth', '*.png')))[::stride]
    else:    
        depth_list = sorted(glob.glob(os.path.join(datapath, 'depth', '*.png')))[::stride]

    for t, (image_file, depth_file) in enumerate(zip(image_list, depth_list)):
        image = cv2.imread(image_file)

        if pred_depth:
            depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH) / 1000.0
        else:
            depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH) / 5000.0

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)
        
        depth = torch.as_tensor(depth)
        depth = F.interpolate(depth[None,None], (h1, w1)).squeeze()
        depth = depth[:h1-h1%8, :w1-w1%8]

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        if use_depth:
            yield t, image[None], depth, intrinsics

        else:
            yield t, image[None], intrinsics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=1024)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--filter_thresh", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--keyframe_thresh", type=float, default=3.5)
    parser.add_argument("--frontend_thresh", type=float, default=16.0)
    parser.add_argument("--frontend_window", type=int, default=16)
    parser.add_argument("--frontend_radius", type=int, default=1)
    parser.add_argument("--frontend_nms", type=int, default=0)

    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--depth", action="store_true")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--csv_suffix", default="")
    parser.add_argument("--dataset_name", default="") #will only be used for csv log file naming
    parser.add_argument("--mono", action="store_true")
    parser.add_argument("--pred_depth", action="store_true")
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    print("Running evaluation on {}".format(args.datapath))
    print(args)

    # this can usually be set to 2-3 except for "camera_shake" scenes
    # set to 2 for test scenes
    stride = 1

    tstamps = []
    for (t, image, depth, intrinsics) in tqdm(image_stream(args.datapath, use_depth=True, stride=stride, pred_depth=args.pred_depth)):
        if not args.disable_vis:
            show_image(image[0])

        if t == 0:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        if args.mono:
            droid.track(t, image, intrinsics=intrinsics)
        else:    
            droid.track(t, image, depth, intrinsics=intrinsics)
    
    traj_est = droid.terminate(image_stream(args.datapath, use_depth=False, stride=stride))

    ### run evaluation ###

    print("#"*20 + " Results...")

    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation

    image_path = os.path.join(args.datapath, 'rgb')
    images_list = sorted(glob.glob(os.path.join(image_path, '*.png')))[::stride]
    tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.array(tstamps))

    gt_file = os.path.join(args.datapath, 'groundtruth.txt')
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=False)
    #import pdb; pdb.set_trace()   
    print(result)

    def store_result_in_csv(sequence_name, rmse_error, csv_suffix, mono):
        mono= "_mono" if mono else ""
        # Define the path to the CSV file
        os.makedirs("inference_logs",exist_ok=True)
        path = f"inference_logs/{args.dataset_name}_ATE_inference_log_{csv_suffix}{mono}.csv"

        # Check if the file exists, create a new dataframe if it doesn't
        if os.path.exists(path):
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame(columns=["Time", "Sequence name", "RMSE error"])

        # Add a new row to the dataframe
        row = pd.DataFrame({"Time": [datetime.now()], "Sequence name": [sequence_name], "RMSE error": [round(rmse_error,6)], "Modality": ["RGB" if mono else "RGBD"], "Weights": [os.path.basename(args.weights)]})
        # df = df.append(row, ignore_index=True)
        df = pd.concat([df, row], ignore_index=True)

        # Write the updated dataframe to the CSV file
        df.to_csv(path, index=False)
    
    store_result_in_csv(os.path.basename(args.datapath), result.stats['rmse'], args.csv_suffix, args.mono)