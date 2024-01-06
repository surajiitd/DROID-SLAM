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

#import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(datapath, modality, stride,for_traj_filling,is_sync):
    """ image generator... this will always give depth also. """

    fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3

    K_l = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3,3)
    d_l = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])

    # read all png images in folder
    images_list = sorted(glob.glob(os.path.join(datapath, 'rgb'+is_sync, '*.png')))[::stride]
    if modality == 'p-rgbd':
        depth_list = sorted(glob.glob(os.path.join(datapath, 'pixelformer_depth', '*.png')))[::stride]
    else:
        depth_list = sorted(glob.glob(os.path.join(datapath, 'depth'+is_sync, '*.png')))[::stride]

    for t, (imfile, depth_file) in enumerate(zip(images_list,depth_list)):
        image = cv2.imread(imfile)
        ht0, wd0, _ = image.shape
        image = cv2.undistort(image, K_l, d_l)
        image = cv2.resize(image, (320+32, 240+16))
        image = torch.from_numpy(image).permute(2,0,1)

        if modality == 'p-rgbd':
            # I have saved pixelformer's depth by multiplying with 1000
            depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH) / 1000.0
        else:
            # the gt depth is 5000 times the depth in meters.
            depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH) / 5000.0
        
        depth = torch.as_tensor(depth)
        depth = F.interpolate(depth[None,None], (240+16, 320+32)).squeeze()
        depth = depth[8:-8, 16:-16]

        intrinsics = torch.as_tensor([fx, fy, cx, cy]).cuda()
        intrinsics[0] *= image.shape[2] / 640.0
        intrinsics[1] *= image.shape[1] / 480.0
        intrinsics[2] *= image.shape[2] / 640.0
        intrinsics[3] *= image.shape[1] / 480.0

        # crop image to remove distortion boundary
        intrinsics[2] -= 16
        intrinsics[3] -= 8
        image = image[:, 8:-8, 16:-16]

        if for_traj_filling:
            yield t, image[None], intrinsics
        else:
            yield t, image[None], depth, intrinsics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--filter_thresh", type=float, default=1.75)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=2.25)
    parser.add_argument("--frontend_thresh", type=float, default=12.0)
    parser.add_argument("--frontend_window", type=int, default=25)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=15.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--csv_suffix", default="")
    parser.add_argument("--dataset_name", default="xyz") #will only be used for csv log file naming
    # parser.add_argument("--mono", action="store_true")
    # parser.add_argument("--pred_depth", action="store_true")
    parser.add_argument("--modality", default="rgb")
    parser.add_argument("--stride", type=int)
    parser.add_argument("--sync", action="store_true")

    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    print("Running evaluation on {}".format(args.datapath))
    print(args)

    droid = Droid(args)
    time.sleep(5)
    stride = args.stride
    is_sync = "_sync" if args.sync else ""
    tstamps = []
    for (t, image, depth, intrinsics) in tqdm(image_stream(args.datapath,args.modality, stride=stride,for_traj_filling=False, is_sync=is_sync)):
        if not args.disable_vis:
            show_image(image)
        if t == 0:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        if args.modality == "rgb":
            droid.track(t, image, intrinsics=intrinsics)
        else:
            droid.track(t, image, depth, intrinsics=intrinsics)


    traj_est = droid.terminate(image_stream(args.datapath, args.modality,stride=stride, for_traj_filling=True, is_sync=is_sync))

    ### run evaluation ###

    print("#"*20 + " Results...")

    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation

    image_path = os.path.join(args.datapath, 'rgb'+is_sync)
    images_list = sorted(glob.glob(os.path.join(image_path, '*.png')))[::2]
    tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.array(tstamps))

    gt_file = os.path.join(args.datapath, 'groundtruth.txt')
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)


    print(result)

    def store_result_in_csv(args, sequence_name, rmse_error):
        # Define the path to the CSV file
        os.makedirs("inference_logs",exist_ok=True)
        args.csv_suffix = args.csv_suffix + is_sync
        path = f"inference_logs/{args.dataset_name}_ATE_{args.modality}_s={args.stride}{args.csv_suffix}.csv"

        # Check if the file exists, create a new dataframe if it doesn't
        if os.path.exists(path):
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame(columns=["Time", "Sequence name", "RMSE error"])

        # Add a new row to the dataframe
        row = pd.DataFrame({"Time": [datetime.now()], "Sequence name": [sequence_name], "RMSE error": [round(rmse_error,6)]})
        # df = df.append(row, ignore_index=True)
        df = pd.concat([df, row], ignore_index=True)

        # Write the updated dataframe to the CSV file
        df.to_csv(path, index=False)

    store_result_in_csv(args, os.path.basename(args.datapath), result.stats['rmse'])
