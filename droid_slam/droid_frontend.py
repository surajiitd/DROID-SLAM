import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph


class DroidFrontend:
    def __init__(self, net, video, args):
        self.video = video  # don't create new object, use the same one as in depth_video
        self.update_op = net.update
        self.graph = FactorGraph(
            video, net.update, max_factors=48, upsample=args.upsample)

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontent variables
        self.is_initialized = False
        self.count = 0

        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2

        self.warmup = args.warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius

    def __update(self):
        """ add edges, perform update """

        self.count += 1    # =1 when first time update is called. # not used anywhere else. it'll count how many times _update() is called
        self.t1 += 1    # =9 when first time update is called.(= warmup + 1)
        # so t1 = 9 , but it's index would be 8. 
        # So current frame is t1-1.

        # Remove any old edges in the graph if they exist
        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        # Add proximity edges to the graph based on the current frame
        # it also uses poses and disps inside distance() function inside it.
        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0),
                                         rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, beta=self.beta, remove=True)

        # INITIALIZE the disparity value for the current frame as sensor's disparity where it is valid.
        self.video.disps[self.t1-1] = torch.where(self.video.disps_sens[self.t1-1] > 0,
                                                  self.video.disps_sens[self.t1-1], self.video.disps[self.t1-1])

        # Perform the first set of updates on the graph
        for itr in range(self.iters1):
            # update the edge's pose and disparity of every edge in the graph.
            self.graph.update(None, None, use_inactive=True)

        
        # set initial pose for next frame
        # poses = SE3(self.video.poses)
        
        # Check if the current frame is a keyframe or not So in distance() func passing ii and jj as just one one element.
        d = self.video.distance(
            [self.t1-3], [self.t1-2], beta=self.beta, bidirectional=True)

        # If the previous frame is not a keyframe, remove it from the graph
        if d.item() < self.keyframe_thresh: 
            #while adding (t1-2)th frame, it was a keyframe. 
            # but after some updates, 
            # it's not fulfilling the same criteria of >= keframe_thresh.
            # instead it is < keyframe_thresh. so remove it.
            self.graph.rm_keyframe(self.t1 - 2)

            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1

        else:
            # Perform the second set of updates on the graph
            for itr in range(self.iters2):
                self.graph.update(None, None, use_inactive=True)

        # set pose for next itration
        """DEBUG: Is (t1-1) is previous frame or current frame? Ans: current frame"""
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

        # update visualization
        self.video.dirty[self.graph.ii.min():self.t1] = True

    def __initialize(self):
        """ initialize the SLAM system """

        self.t0 = 0 # start of the window
        self.t1 = self.video.counter.value # end of the window
        # self.video.counter.value is 1-based indexing (means this is = number of keyframes I have got till now) And current index is (t1-1)

        """ add edges between neighboring frames within radius r """
        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        """
        add_proximity_factors(): add edges to the factor graph based on proximity radius 
        then based on distance(means add edges one by one 
        in sorted order till no. of edges is <= self.max_factors """
        self.graph.add_proximity_factors(
            0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)


        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1-1].clone()
        self.last_disp = self.video.disps[self.t1-1].clone()
        self.last_time = self.video.tstamp[self.t1-1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1

            # update visualization
            self.video.dirty[:self.t1] = True

        # explaned by chatgpt: This line is removing edges that were added in the earlier initialization stages and 
        # are no longer needed, to keep the number of factors in the factor graph under control. 
        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)

    def __call__(self):
        """ main update """

        # do initialization after getting enough frames (= warmup = 8 here)
        if not self.is_initialized and self.video.counter.value == self.warmup:
            # __initialize() will be called after we get self.warmup keyframes
            self.__initialize()

        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            # __update() will be called for every new keyframe after initialisation.
            self.__update()
