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


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def print_minmax(arr,desc):
    """visualize depths and uncertainty of any method"""
    
    print("*" * 60)
    print("***{}***  :".format(desc))
    print("arr.shape = {}".format(arr.shape))
    print("type(arr[0,0] = {}".format(type(arr[0,0])))
    print("np.min = {}".format(np.min(arr)))
    print("np.max = {}".format(np.max(arr)))
    print("np.mean = {}".format(np.mean(arr)))
    print("np.median = {}".format(np.median(arr)))
    #print("arr[200:220,200:220] = \n",arr[200:220,200:220])
    print("arr[0:10,0:10] = \n",arr[0:10,0:10])
    print("*" * 60 + "\n")


def read_groundtruth(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    gt_poses_list = []
    lines = lines[1:]
    for line in lines:
        
        if line.startswith('#'):
            gt_poses_list.append([0, 0, 0, 0, 0, 0, 1])
            continue
        data = line.split()
        #timestamp = int(data[0])
        tx, ty, tz = [float(x) for x in data[0:3]]
        qx, qy, qz, qw = [float(x) for x in data[3:]]
        gt_poses_list.append([tx, ty, tz, qx, qy, qz, qw])
    
    print("# of gt poses = ", len(gt_poses_list))
    return gt_poses_list



def image_stream(datapath, use_depth=False, stride=1, use_pred_depth = False):
    """ image generator """
    fx,fy,cx,cy = 585,585,320,240 #wrong mostly ...given in website for default intrinsics for the depth camera
    #fx, fy, cx, cy = np.loadtxt(os.path.join(datapath, 'calibration.txt')).tolist()
    image_list = sorted(glob.glob(os.path.join(datapath, '*color.png')))[::stride]
    if use_pred_depth:
        max_depth_threshold = ""
        depth_list = sorted(glob.glob(os.path.join(datapath, f'pixelformer_depth{max_depth_threshold}', '*depth.png')))[::stride]
    else:    
        depth_list = sorted(glob.glob(os.path.join(datapath, '*depth.png')))[::stride]
    gt_depth_list = sorted(glob.glob(os.path.join(datapath, '*depth.png')))[::stride]
    print(" # of images and depths = ", len(image_list), len(depth_list))
    # gt_poses_list = read_groundtruth(os.path.join(args.datapath, "groundtruth_exact_in_number.txt"))
    gt_poses_list = read_groundtruth(os.path.join(args.datapath, "groundtruth.txt"))

    for t, (image_file, depth_file, gt_pose, gt_depth_file) in enumerate(zip(image_list, depth_list,gt_poses_list, gt_depth_list)):
        #print("t = ",t)    
        image = cv2.imread(image_file)
        
        height,width,_= image.shape
        #image = image[height%32:, width%32:,:]
        #image = cv2.resize(image, (width, height))

        if use_pred_depth:
            depth = cv2.imread(depth_file, -1) / 1000.0
            
            #if t==0:
            #   print("initialising depth with 0")
            #depth = depth * 0.
            #print_minmax(depth,"pred_depth")
            #sys.exit(0)
        else:
            depth = cv2.imread(depth_file, -1) / 1000.0  # cv2.IMREAD_ANYDEPTH
            #print_minmax(depth,"gt_depth")
            #sys.exit(0)
        gt_depth = cv2.imread(gt_depth_file, -1) / 1000.0  # cv2.IMREAD_ANYDEPTH
            

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        # image = cv2.resize(image, (w1, h1))
        # image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)
        
        depth = torch.as_tensor(depth)
        # depth = F.interpolate(depth[None,None], (h1, w1)).squeeze()
        # depth = depth[:h1-h1%8, :w1-w1%8]

        """flow experiment start"""
        gt_depth = torch.as_tensor(gt_depth)
        # gt_depth = F.interpolate(gt_depth[None,None], (h1, w1)).squeeze()
        # gt_depth = gt_depth[:h1-h1%8, :w1-w1%8]

        gt_pose = torch.as_tensor(gt_pose)
        """flow experiment end"""

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        # intrinsics[0::2] *= (w1 / w0)
        # intrinsics[1::2] *= (h1 / h0)


        if use_depth:
            yield t, image[None], depth, intrinsics, gt_pose, gt_depth

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
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    parser.add_argument("--without_depth", action="store_true")
    parser.add_argument("--pixelformer", action="store_true")
    

    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')
    print("\n\n")
    print("Running evaluation on {}".format(args.datapath),end="\n\n")
    print("args = ",args,end="\n\n")

    # this can usually be set to 2-3 except for "camera_shake" scenes
    # set to 2 for test scenes
    stride = 1

    tstamps = []
    for (t, image, depth, intrinsics, gt_pose, gt_depth) in tqdm(image_stream(args.datapath, use_depth=True, stride=stride, use_pred_depth = args.pixelformer)):
        if not args.disable_vis:
            show_image(image[0])

        if t == 0:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        if args.without_depth:
            
            droid.track(t, image, intrinsics=intrinsics, gt_pose = gt_pose)
        else:
            # print("t = ",t)    
            droid.track(t, image, depth, intrinsics=intrinsics, gt_pose = gt_pose, gt_depth = gt_depth)

    traj_est = droid.terminate(image_stream(args.datapath, use_depth=False, stride=stride))



    ### run evaluation ###
    if args.without_depth:
        mode = "Without depth!! Just using RGB"
    elif args.pixelformer:
        mode = "Using predicted depth"
    else: 
        mode = "Using GT depth"
    print("#"*20 + " Evaluation mode: " + mode + "#"*20)


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

    #import pdb; pdb.set_trace()
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True) #originally correct_scale was False

    #print(result.stats)
    print(result)




