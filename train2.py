import sys
sys.path.append('droid_slam')

import cv2
import numpy as np
#added by suraj to romove error: "AttributeError: module 'numpy.random' has no attribute 'default_rng'"
# np.random.bit_generator = np.random._bit_generator

from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_readers.factory import dataset_factory

from lietorch import SO3, SE3, Sim3
from geom import losses
# from geom.losses import geodesic_loss, residual_loss, flow_loss
from geom.graph_utils import build_frame_graph

# network
from droid_net import DroidNet
#from logger import Logger

# DDP training
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(gpu, args):
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',     
    	world_size=args.world_size,                              
    	rank=gpu)

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def train(gpu, args):
    """ Test to make sure project transform correctly maps points """

    # coordinate multiple GPUs
    setup_ddp(gpu, args)
    # rng: random number generator
    rng = np.random.default_rng(12345)

    N = args.n_frames
    model = DroidNet()
    model.cuda()
    model.train()

    model = DDP(model, device_ids=[gpu], find_unused_parameters=False)

    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))

    # fetch dataloader
    db = dataset_factory(['tartan'], datapath=args.datapath, n_frames=args.n_frames, fmin=args.fmin, fmax=args.fmax)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        db, shuffle=True, num_replicas=args.world_size, rank=gpu)

    train_loader = DataLoader(db, batch_size=args.batch, sampler=train_sampler, num_workers=2)

    # fetch optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps, pct_start=0.01, cycle_momentum=False)

    #logger = Logger(args.name, scheduler)
    should_keep_training = True
    total_steps = 0

    while should_keep_training:
        for i_batch, item in enumerate(train_loader):
            optimizer.zero_grad()

            images, poses, disps, intrinsics = [x.to('cuda') for x in item]
            """DEBUG Training: when i_batch = 0 (1st batch)
            images.shape = torch.Size([1, 7, 3, 384, 512])
            poses.shape =  torch.Size([1, 7, 7])
            disps.shape = torch.Size([1, 7, 384, 512])
            intrinsics.shape = torch.Size([1, 7, 4])
            So, I have 7 fames in every batch and I am doing 15 updates to depths & poses per batch 
            inside model()'s forward() function.
            """
            # convert poses w2c -> c2w
            Ps = SE3(poses).inv()
            Gs = SE3.IdentityLike(Ps)

            # randomize frame graph
            if np.random.rand() < 0.5:
                graph = build_frame_graph(poses, disps, intrinsics, num=args.edges)       
            else:
                graph = OrderedDict()
                for i in range(N):
                    graph[i] = [j for j in range(N) if i!=j and abs(i-j) <= 2]
            
            # fix first two camera poses
            Gs.data[:,0] = Ps.data[:,0].clone()
            Gs.data[:,1:] = Ps.data[:,[1]].clone()
            disp0 = torch.ones_like(disps[:,:,3::8,3::8])

            # perform random restarts
            r = 0
            while r < args.restart_prob:
                r = rng.random()
                
                intrinsics0 = intrinsics / 8.0
                poses_est, disps_est, residuals = model(Gs, images, disp0, intrinsics0, graph, num_steps=args.iters, fixedp=2)
                # poses_est is a list of 15 length with each element of poses_est[0].tangent_shape torch.Size([1, 7, 6])
                # disps_est is a list of 15 length with each element of disps_est[0].shape torch.Size([1, 7, 384, 512])
                # residuals is a list of 15 length with each element of residuals[0].shape torch.Size([1, 24, 48, 64, 2])
                """ WHAT IS 15 HERE??? => 15 is no. of iterations of convGRU update inside model() function"""
                """ and 7 is #nodes for sure and 24 would be #edges in the graph as the size of ii and jj is also 24"""

                geo_loss, geo_metrics = losses.geodesic_loss(Ps, poses_est, graph, do_scale=False)
                res_loss, res_metrics = losses.residual_loss(residuals)
                flo_loss, flo_metrics = losses.flow_loss(Ps, disps, poses_est, disps_est, intrinsics, graph)

                loss = args.w1 * geo_loss + args.w2 * res_loss + args.w3 * flo_loss
                loss.backward()

                Gs = poses_est[-1].detach()
                disp0 = disps_est[-1][:,:,3::8,3::8].detach()

            metrics = {}
            metrics.update(geo_metrics)
            metrics.update(res_metrics)
            metrics.update(flo_metrics)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            
            total_steps += 1

            if gpu == 0:
                
                #logger.push(metrics)
                pass

            if total_steps % 10000 == 0 and gpu == 0:
                PATH = 'checkpoints/%s_%06d.pth' % (args.name, total_steps)
                torch.save(model.state_dict(), PATH)

            if total_steps >= args.steps:
                should_keep_training = False
                break

    dist.destroy_process_group()
                

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--datasets', nargs='+', help='lists of datasets for training')
    parser.add_argument('--datapath', default='datasets/TartanAir', help="path to dataset directory")
    parser.add_argument('--gpus', type=int, default=4)

    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--iters', type=int, default=15)
    parser.add_argument('--steps', type=int, default=250000)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--clip', type=float, default=2.5)
    parser.add_argument('--n_frames', type=int, default=7)

    parser.add_argument('--w1', type=float, default=10.0)
    parser.add_argument('--w2', type=float, default=0.01)
    parser.add_argument('--w3', type=float, default=0.05)

    parser.add_argument('--fmin', type=float, default=8.0)
    parser.add_argument('--fmax', type=float, default=96.0)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--edges', type=int, default=24)
    parser.add_argument('--restart_prob', type=float, default=0.2)

    args = parser.parse_args()

    args.world_size = args.gpus
    print(args)

    import os
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    args = parser.parse_args()
    args.world_size = args.gpus

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    mp.spawn(train, nprocs=args.gpus, args=(args,))

