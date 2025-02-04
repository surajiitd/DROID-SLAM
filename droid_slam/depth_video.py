import numpy as np
import torch
import lietorch
import droid_backends

from torch.multiprocessing import Process, Queue, Lock, Value
from collections import OrderedDict

from droid_net import cvx_upsample
import geom.projective_ops as pops


class DepthVideo:
    def __init__(self, image_size=[480, 640], buffer=1024, stereo=False, device="cuda:0"):

        # current keyframe count
        self.counter = Value('i', 0)
        self.ready = Value('i', 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]

        ### state attributes ###
        self.tstamp = torch.zeros(
            buffer, device="cuda", dtype=torch.float).share_memory_()
        self.images = torch.zeros(
            buffer, 3, ht, wd, device="cuda", dtype=torch.uint8)
        self.dirty = torch.zeros(
            buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.red = torch.zeros(buffer, device="cuda",
                               dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(
            buffer, 7, device="cuda", dtype=torch.float).share_memory_()
        self.disps = torch.ones(
            buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(
            buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(
            buffer, ht, wd, device="cuda", dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(
            buffer, 4, device="cuda", dtype=torch.float).share_memory_()

        self.stereo = stereo

        # by Suraj
        self.gt_poses = torch.zeros(
            buffer, 7, device="cuda", dtype=torch.float).share_memory_()
        self.gt_disps = torch.ones(
            buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()

        c = 1 if not self.stereo else 2

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, c, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//8, wd//8,dtype=torch.half, device="cuda").share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//8, wd//8,dtype=torch.half, device="cuda").share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor(
            [0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")

    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        """
        set all data attributes for a given key-frame index. if it is a new index, 
        then increase the counter.value of video by 1. 
        else if it is an existing index, then just overwrite the data attributes.
        """

        if isinstance(index, int) and index >= self.counter.value:
            # add new key-frame to video and increase the counter.value of video by 1.
            self.counter.value = index + 1

        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1
        #print("\tself.counter.value = ",self.counter.value)

        # self.dirty[index] = True
        self.tstamp[index] = item[0]
        self.images[index] = item[1]

        # compare it's size with gt_pses...
        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            depth = item[4][3::8, 3::8]
            self.disps_sens[index] = torch.where(depth > 0, 1.0/depth, depth)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            self.nets[index] = item[7]

        if len(item) > 8:
            self.inps[index] = item[8]

        # by Suraj
        if len(item) > 9 and item[9] is not None:
            self.gt_poses[index] = item[9]

        if len(item) > 10 and item[10] is not None:
            gt_depth = item[10][3::8, 3::8]
            self.gt_disps[index] = torch.where(
                gt_depth > 0, 1.0/gt_depth, gt_depth)

    def __setitem__(self, index, item):
        with self.get_lock():   # this fun is nothing but "return self.counter.get_lock()"
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing: -1 is the last key-frame... so on and so forth
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)

    ### geometric operations ###
    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """
        
        #upsample pixel-wise transformation field
        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def normalize(self):
        """ normalize depth and poses """
        """DEBUG: check what kind of normalization is this and why is it needed?"""
        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value, :3] *= s
            self.dirty[:self.counter.value] = True

    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(
                Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def reproject_with_gt_pose_and_depth(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)

        # current frame index
        curr = self.counter.value
        gt_pose = self.gt_poses[curr-1]
        gt_disp = self.gt_disps[curr-1]

        # copy the list of poses and disps
        new_poses = self.poses.detach().clone()
        new_disps = self.disps.detach().clone()
        new_intrinsics = self.intrinsics.detach().clone()

        # replace current frame's pose and disp with gt_pose and gt_disp.
        new_poses[curr-1] = gt_pose
        new_disps[curr-1] = gt_disp

        Gs = lietorch.SE3(new_poses[None])

        coords, valid_mask = \
            pops.projective_transform(
                Gs, new_disps[None], new_intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """
        """ if ii is None: 
                distance b/w all(=self.counter.value) frames in the video. So it'll compute a NxN matrix. return_matrix=True in that case.
            else:
                distance b/w the frames specified by ii and jj. It'll compute a 1d tensor of length len(ii). return_matrix=False in that case.
        
        ###called a CUDA kernel.###
        In paper they said that distance is mean optical ﬂow b/w those 2 frames.
        """
        #print("ii and jj are: ", ii, jj)
        return_matrix = False  # otherwise it'll return a 1d tensor.
        if ii is None:
            return_matrix = True
            N = self.counter.value
            #print("Making NxN matrix, with N = self.counter.value = ", N)
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))

        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)
        #print("d.shape =", d.shape, "ii.shape = ", ii.shape)
        return d

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False):
        """ 
        # eta: it is "damping" from where this fun is called(factor_graph's update()).
        # t0: start of the window
        # t1: end of the window
        # itrs: no. of iterations of BA
        # lm, ep : these parameters are only used in A.solve(lm,ep) where A is an object of SparseBlock class
        # and the way they are used in eqn(inside solve() of SparseBlock class) are: 
        # "L.diagonal().array() += ep + lm * L.diagonal().array();"
        
        dense bundle adjustment (DBA) 
        ###called a CUDA kernel.###
        """

        with self.get_lock():

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            # call python ba code here.


            droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.disps_sens,
                                target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)

            self.disps.clamp_(min=0.001)
