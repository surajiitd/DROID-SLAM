import torch
import lietorch
import numpy as np

import matplotlib.pyplot as plt
from lietorch import SE3
from modules.corr import CorrBlock, AltCorrBlock
import geom.projective_ops as pops



class FactorGraph:
    def __init__(self, video, update_op, device="cuda:0", corr_impl="volume", max_factors=-1, upsample=False):
        self.video = video
        self.update_op = update_op
        self.device = device
        self.max_factors = max_factors
        self.corr_impl = corr_impl
        self.upsample = upsample

        # operator at 1/8 resolution
        self.ht = ht = video.ht // 8
        self.wd = wd = video.wd // 8

        self.coords0 = pops.coords_grid(ht, wd, device=device)
        
        """DEBUG: what is ii and jj here? ans: (mp)current set of keyframes that are active(that are in the factor graph)"""
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        self.corr, self.net, self.inp = None, None, None
        self.damping = 1e-6 * torch.ones_like(self.video.disps)

        self.target = torch.zeros(
            [1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight = torch.zeros(
            [1, 0, ht, wd, 2], device=device, dtype=torch.float)

        # inactive factors
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.ii_bad = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_bad = torch.as_tensor([], dtype=torch.long, device=device)

        self.target_inac = torch.zeros(
            [1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight_inac = torch.zeros(
            [1, 0, ht, wd, 2], device=device, dtype=torch.float)

    def __filter_repeated_edges(self, ii, jj):
        """ remove duplicate edges """  

        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set(
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)] +
            [(i.item(), j.item()) for i, j in zip(self.ii_inac, self.jj_inac)])

        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset

        return ii[keep], jj[keep]

    def print_edges(self):
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        ix = np.argsort(ii)
        ii = ii[ix]
        jj = jj[ix]

        w = torch.mean(self.weight, dim=[0, 2, 3, 4]).cpu().numpy()
        w = w[ix]
        for e in zip(ii, jj, w):
            print(e)
        print()

    def filter_edges(self):
        """ remove bad edges """
        conf = torch.mean(self.weight, dim=[0, 2, 3, 4])
        mask = (torch.abs(self.ii-self.jj) > 2) & (conf < 0.001)

        self.ii_bad = torch.cat([self.ii_bad, self.ii[mask]])
        self.jj_bad = torch.cat([self.jj_bad, self.jj[mask]])
        self.rm_factors(mask, store=False)

    def clear_edges(self):
        self.rm_factors(self.ii >= 0)
        self.net = None
        self.inp = None
    @torch.cuda.amp.autocast(enabled=True) #True: It allows computations to be performed in lower-precision formats for improved performance. 
    def add_factors(self, ii, jj, remove=False): # OR add_edges()
        """ add edges to factor graph """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # remove duplicate edges
        ii, jj = self.__filter_repeated_edges(ii, jj)

        if ii.shape[0] == 0:
            return
        """place limit on number of factors:
        If there is a limit on the number of edges in the graph, 
        and the number of edges plus the new edges would exceed this limit, 
        then remove some existing edges so that the limit is not exceeded.
        """
        # will not go inside this while initialisation of frontend ... because remove == False ... but will go inside in update() of frontend.
        if self.max_factors > 0 and self.ii.shape[0] + ii.shape[0] > self.max_factors \
                and self.corr is not None and remove:

            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        net = self.video.nets[ii].to(self.device).unsqueeze(0)  # self.video.nets.shape = torch.Size([512, 128, 48, 64]) ... But we take duplicates now (by indexing using ii)

        # save correlation pyramids for new edges
        if self.corr_impl == "volume": # so in backend self.corr and self.inp tensors are not created.
            # this section will not get called in Backend (corr_impl="alt") ... So no memory for correlation vol is used in backend.
            c = (ii == jj).long()
            fmap1 = self.video.fmaps[ii, 0].to(self.device).unsqueeze(0)  # uses 27MB for 36 edges' fmap
            fmap2 = self.video.fmaps[jj, c].to(self.device).unsqueeze(0) # same...

            # correlation pyramid for all the new edges(frame pairs) is computed using just below line. 
            # one dim is reserved for which edge it is. ex: corr-pyramid[0]'s size was [1, 36, 48, 64, 48, 64] 
            # when ii and jj's shape here is : torch.Size([36]) when I first enter this line of code.
            corr = CorrBlock(fmap1, fmap2)
            
            # print(f"\nAdding {fmap1.shape[1]} Edges in graph") #fmap1.shape = torch.Size([1, 36, 128, 48, 64]) when 36 new edges are getting added in graph.
            # append correlation volumes for new edges in self.corr
            #DEBUG: check for how many edges does it store corr-pyramid with increasing #frames. $ print(self.corr.corr_pyramid[0].shape)
            self.corr = corr if self.corr is None else self.corr.cat(corr)
            #current_edges = self.corr.corr_pyramid[0].shape[0]  # current_edges in graph.
            # print(f"Now {current_edges} Edges are in graph",)
            
            # append inp for new edges in self.inp
            inp = self.video.inps[ii].to(self.device).unsqueeze(0)
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)

        with torch.cuda.amp.autocast(enabled=False): #False: means full precision...It doesn't allows lower-precision computations.

            # target is DCF for all the new edges: 
            # ex: target's size = torch.Size([1, 36, 48, 64, 2])
            #     when ii and jj's shape here is : torch.Size([36]) when I first enter this line of code.
            target, _ = self.video.reproject(ii, jj)
            
            # weight's size = torch.Size([1, 36, 48, 64, 2])
            weight = torch.zeros_like(target)
        
        #append ii(starting node) and jj(target node) for new edges in self.ii and self.jj
        # if self.corr_impl != "volume": 
        #     print(f"\nAdding {ii.shape[0]} Edges in graph")

        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)

        # if self.corr_impl != "volume": 
        #     print(f"Now {self.ii.shape[0]} Edges are in graph",)

        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        # reprojection factors
        self.net = net if self.net is None else torch.cat([self.net, net], 1)

        #append target and weight for new edges.
        self.target = torch.cat([self.target, target], 1)
        self.weight = torch.cat([self.weight, weight], 1)


    @torch.cuda.amp.autocast(enabled=True)
    def rm_factors(self, mask, store=False):
        """ drop edges from factor graph """
        """DEBUG: check mask's size"""
        """ according to my understanding, mask's size should be equal to the no. of edges in the graph.
        and it is True for all edges that need to be removed."""
        
        # store estimated factors (OR edges)
        if store:
            self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], 0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], 0)
            self.target_inac = torch.cat(
                [self.target_inac, self.target[:, mask]], 1)
            self.weight_inac = torch.cat(
                [self.weight_inac, self.weight[:, mask]], 1)
        
        # print(f"\nDeleting {np.sum(mask.clone().detach().cpu().numpy())} edges.")
        # print(f"Now {np.sum(~mask.clone().detach().cpu().numpy())} edges are in graph.")
        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]

        if self.corr_impl == "volume": # if frontend(corr_impl="volume") then only need to clear corr volume, otherwise there is not self.corr variable itself in backend.
            self.corr = self.corr[~mask]

        if self.net is not None:
            self.net = self.net[:, ~mask]

        if self.inp is not None: # it is None in backend
            self.inp = self.inp[:, ~mask]

        self.target = self.target[:, ~mask]
        self.weight = self.weight[:, ~mask]

        # added by me to see if the correlation-volume for deleted edges was remaining still in memory...
        #torch.cuda.empty_cache()


    @torch.cuda.amp.autocast(enabled=True)
    def rm_keyframe(self, ix):
        """ drop edges from factor graph """

        with self.video.get_lock():
            self.video.images[ix] = self.video.images[ix+1]
            self.video.poses[ix] = self.video.poses[ix+1]
            self.video.disps[ix] = self.video.disps[ix+1]
            self.video.disps_sens[ix] = self.video.disps_sens[ix+1]
            self.video.intrinsics[ix] = self.video.intrinsics[ix+1]

            self.video.nets[ix] = self.video.nets[ix+1]
            self.video.inps[ix] = self.video.inps[ix+1]
            self.video.fmaps[ix] = self.video.fmaps[ix+1]

        # if an edge will have either of the nodes as the keyframe, then that edge will be removed.
        # So return True for that edge in m.
        """DEBUG: check m's size... I am hoping that it is equal to the no. of edges in the graph."""
        m = (self.ii_inac == ix) | (self.jj_inac == ix)

        # to make the indices of the inactive lists coherent with actual list indices.
        self.ii_inac[self.ii_inac >= ix] -= 1
        self.jj_inac[self.jj_inac >= ix] -= 1

        if torch.any(m):
            # also removing that keyframe from inactive lists. (by srj)
            self.ii_inac = self.ii_inac[~m]
            self.jj_inac = self.jj_inac[~m]
            self.target_inac = self.target_inac[:, ~m]
            self.weight_inac = self.weight_inac[:, ~m]

        # Similarly remove edges from active lists also.
        m = (self.ii == ix) | (self.jj == ix)

        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1
        self.rm_factors(m, store=False)


    @torch.cuda.amp.autocast(enabled=True)
    def update(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, motion_only=False):
        """ run update operator on whole factor graph (covisibility graph) """
        # t0: start of the window
        # t1: end of the window
        # itrs: number of BA iterations.
        # use_inactive: if True, then use inactive edges also for BA.
        # EP: small value to avoid division by zero.
        # motion features
        with torch.cuda.amp.autocast(enabled=False):
            coords1, mask = self.video.reproject(self.ii, self.jj) # project points from ii -> jj 
            #print(f"called graph.update for current frame = {self.video.counter.value}")
            #coords1, mask = self.video.reproject_with_gt_pose_and_depth(self.ii, self.jj) # project points from ii -> jj 
            motn = torch.cat(
                [coords1 - self.coords0, self.target - coords1], dim=-1)
            motn = motn.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

        # correlation features
        corr = self.corr(coords1)
        

        # estimate flow-revisions for each edge of the graph. 
        # [ as self.ii and self.jj are the indices of the edges and it represents the edges of the graph.]
        self.net, delta, weight, damping, upmask = \
            self.update_op(self.net, self.inp, corr, motn, self.ii, self.jj)  # update function of updateModule class

        if t0 is None:
            t0 = max(1, self.ii.min().item()+1)

        with torch.cuda.amp.autocast(enabled=False):
            self.target = coords1 + delta.to(dtype=torch.float)
            self.weight = weight.to(dtype=torch.float)

            ht, wd = self.coords0.shape[0:2]
            self.damping[torch.unique(self.ii)] = damping # see later what damping is doing.
            
            # BUG: sample
            # TODO: sample
            # NOTE: Inorder to call DBA, we are concatenating all the inactive edges also in 
            # factor graph.
            if use_inactive:
                m = (self.ii_inac >= t0 - 3) & (self.jj_inac >= t0 - 3)
                ii = torch.cat([self.ii_inac[m], self.ii], 0)
                jj = torch.cat([self.jj_inac[m], self.jj], 0)
                target = torch.cat([self.target_inac[:, m], self.target], 1)
                weight = torch.cat([self.weight_inac[:, m], self.weight], 1)

            else:
                ii, jj, target, weight = self.ii, self.jj, self.target, self.weight
            """flow exp start"""
            """DEBUG: check size of target before and after below line."""
            #compute DCF from gt-pose and gt-depth
            #target,_ = self.video.reproject_with_gt_pose_and_depth(self.ii, self.jj)
            """flow exp end"""
            damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP

            target = target.view(-1, ht, wd, 2).permute(0,
                                                        3, 1, 2).contiguous()
            weight = weight.view(-1, ht, wd, 2).permute(0,
                                                        3, 1, 2).contiguous()

            # dense bundle adjustment
            self.video.ba(target, weight, damping, ii, jj, t0, t1,
                          itrs=itrs, lm=1e-4, ep=0.1, motion_only=motion_only)
            if self.upsample:
                self.video.upsample(torch.unique(self.ii), upmask)

        self.age += 1


    @torch.cuda.amp.autocast(enabled=False)
    def update_lowmem(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, steps=8):
        """ run update operator on factor graph - reduced memory implementation """

        # alternate corr implementation
        t = self.video.counter.value

        num, rig, ch, ht, wd = self.video.fmaps.shape

        corr_op = AltCorrBlock(self.video.fmaps.view(1, num*rig, ch, ht, wd))

        for step in range(steps):
            print("Global BA Iteration #{}".format(step+1))
            with torch.cuda.amp.autocast(enabled=False):
                coords1, mask = self.video.reproject(self.ii, self.jj)
                motn = torch.cat(
                    [coords1 - self.coords0, self.target - coords1], dim=-1)
                motn = motn.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

            s = 8
            for i in range(0, self.jj.max()+1, s):
                v = (self.ii >= i) & (self.ii < i + s)
                iis = self.ii[v]
                jjs = self.jj[v]

                ht, wd = self.coords0.shape[0:2]
                corr1 = corr_op(coords1[:, v], rig * iis,
                                rig * jjs + (iis == jjs).long())

                with torch.cuda.amp.autocast(enabled=True):

                    net, delta, weight, damping, upmask = \
                        self.update_op(
                            self.net[:, v], self.video.inps[None, iis], corr1, motn[:, v], iis, jjs)
                    if self.upsample:
                        self.video.upsample(torch.unique(iis), upmask)

                self.net[:, v] = net
                self.target[:, v] = coords1[:, v] + delta.float()
                self.weight[:, v] = weight.float()
                self.damping[torch.unique(iis)] = damping

            damping = .2 * self.damping[torch.unique(self.ii)].contiguous() + EP # EP=1e-7
            target = self.target.view(-1, ht, wd,2).permute(0, 3, 1, 2).contiguous()
            weight = self.weight.view(-1, ht, wd,2).permute(0, 3, 1, 2).contiguous()

            # dense bundle adjustment
            self.video.ba(target, weight, damping, self.ii, self.jj, 1, t, 
                          itrs=itrs, lm=1e-5, ep=1e-2, motion_only=False)

            self.video.dirty[:t] = True


    def add_neighborhood_factors(self, t0, t1, r=3):
        """ add edges between neighboring frames within radius r """

        ii, jj = torch.meshgrid(torch.arange(t0, t1), torch.arange(t0, t1))
        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device)
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device)

        c = 1 if self.video.stereo else 0

        keep = ((ii - jj).abs() > c) & ((ii - jj).abs() <= r)
        self.add_factors(ii[keep], jj[keep])

        
    def add_proximity_factors(self, t0=0, t1=0, rad=2, nms=2, beta=0.25, thresh=16.0, remove=False):
        """ 
        add edges to the factor graph based on proximity radius(that is "make edges to keyframes that are at least >= proximity radius(=3) frames old) then based on distance(means add edges one by one 
        in sorted order till no. of edges is <= self.max_factors..... 
        i.e: if len(es) > self.max_factors:         break
        
        The code for adding based on distance is after line `es = []` 
        """

        # calculate distance b/w all possible pairs of frames(nodes).
        t = self.video.counter.value
        ix = torch.arange(t0, t)
        jx = torch.arange(t1, t)

        ii, jj = torch.meshgrid(ix, jx)
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)
        d = self.video.distance(ii, jj, beta=beta)

        # make adges b/w all **previous** keyframes(from ii) that are at rad(=3). make rest all as infinity.
        d[ii - rad < jj] = np.inf
        d[d > 100] = np.inf

        ii1 = torch.cat([self.ii, self.ii_bad, self.ii_inac], 0)
        jj1 = torch.cat([self.jj, self.jj_bad, self.jj_inac], 0)
        for i, j in zip(ii1.cpu().numpy(), jj1.cpu().numpy()):
            #Non Maximal Suppression: ensures that the node only includes-
            # the neighboring nodes that are not too close (that's why <=) to each other. 
            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0): #ignore max() it is just to ignore negative values)
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):       # i1 and j1 should be in the range
                            # set distance inf for row i1 and column j1 (since d is a flattened array, 
                            # we'll do Row-Major order)
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf   # (t-t1) is # of columns

        # list of edges to be added to the covisibility graph. 
        es = []
        """
        In below code, for each edge addition, we are making it's distance to inf because 
        to prevent the same node or an edge from being added multiple times to the graph.
        By setting the distances to infinity, these edges or self-loops are effectively excluded 
        from consideration in the following sections of the code where edges are added 
        based on the sorted distances. This ensures that the graph only includes unique edges and nodes.
        """
        # This for loop will add edges based on whether the frames are within the proximity radius.
        # so it is similar to add_neighbourhood_edges().
        for i in range(t0, t):
            if self.video.stereo:
                # add a self-loop to node `i`
                es.append((i, i))
                # sets the distance b/w node `i` and itself to inf.
                d[(i-t0)*(t-t1) + (i-t1)] = np.inf

            for j in range(max(i-rad-1, 0), i):
                es.append((i, j))
                es.append((j, i))
                d[(i-t0)*(t-t1) + (j-t1)] = np.inf

        ix = torch.argsort(d)
        # we are traversing and adding edges in sorted order of the distance values of edges.
        for k in ix:
            if d[k].item() > thresh:
                continue

            if len(es) > self.max_factors:
                break

            i = ii[k]
            j = jj[k]

            # bidirectional
            es.append((i, j))
            es.append((j, i))

            #Non Maximal Suppression: ensures that the node only includes-
            # the neighboring nodes that are not too close (that's why <=) to each other. 
            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0): #ignore max(it is just to ignore negative values)
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t): # i1 and j1 should be in the range
                            # set distance inf for row i1 and column j1 (since d is a flattened array, 
                            # we'll do Row-Major order)
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf

        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        self.add_factors(ii, jj, remove)
