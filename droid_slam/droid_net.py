import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from modules.extractor import BasicEncoder
from modules.corr import CorrBlock
from modules.gru import ConvGRU
from modules.clipping import GradientClip

from lietorch import SE3
from geom.ba import BA

import geom.projective_ops as pops
from geom.graph_utils import graph_to_edge_list, keyframe_indicies

from torch_scatter import scatter_mean
from sys import getsizeof


def cvx_upsample(data, mask):
    # cvx_upsample is the same function that is used in RAFT to upsample optical flow.
    # BUT in DROID it is used only to upsample disparity and NOT flow.
    """ upsample pixel-wise transformation field """
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2)
    # mask.shape = torch.Size([1, 576, ht, wd]) where 576 = 9*8*8
    mask = mask.view(batch, 1, 9, 8, 8, ht, wd)
    mask = torch.softmax(mask, dim=2)

    up_data = F.unfold(data, [3,3], padding=1)
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

    up_data = torch.sum(mask * up_data, dim=2) # (mask*up_data).shape=(batch, dim, 9, 8, 8, ht, wd)
    # up_data.shape = (batch, dim, 8, 8, ht, wd) # sum over 9 (9 gayab ho gaya)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1)
    # up_data.shape = (batch, ht, 8, wd, 8, dim) 
    up_data = up_data.reshape(batch, 8*ht, 8*wd, dim) # now (h,w) is 8 times larger than (ht,wd)

    return up_data

def upsample_disp(disp, mask):
    #while training, my upsample_mask is predicted from n/w
    batch, num, ht, wd = disp.shape
    disp = disp.view(batch*num, ht, wd, 1)
    mask = mask.view(batch*num, -1, ht, wd)
    # cvx_upsample is the same function that is used in RAFT to upsample optical flow.
    return cvx_upsample(disp, mask).view(batch, num, 8*ht, 8*wd)


class GraphAgg(nn.Module):
    def __init__(self):
        super(GraphAgg, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.eta = nn.Sequential(
            nn.Conv2d(128, 1, 3, padding=1),
            GradientClip(),
            nn.Softplus())

        self.upmask = nn.Sequential(
            nn.Conv2d(128, 8*8*9, 1, padding=0))

    def forward(self, net, ii):
        """srj: Graph aggregation operator:
        In this case, `ii` is used to group the feature map 
        based on the indices and take mean of each group to 
        aggregate the feature map.
        """
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch*num, ch, ht, wd)

        _, ix = torch.unique(ii, return_inverse=True)
        net = self.relu(self.conv1(net))

        net = net.view(batch, num, 128, ht, wd)
        """main aggregation is happening here"""
        net = scatter_mean(net, ix, dim=1)  
        net = net.view(-1, 128, ht, wd)

        net = self.relu(self.conv2(net))

        eta = self.eta(net).view(batch, -1, ht, wd)
        upmask = self.upmask(net).view(batch, -1, 8*8*9, ht, wd)

        return .01 * eta, upmask


class UpdateModule(nn.Module):
    def __init__(self):
        super(UpdateModule, self).__init__()
        
        # radius = 3, 
        # lookup = 7 x 7  [2*3 + 1 = 7]
        # flatenned lookup = 7**2 = 49
        # 4 is levels of pyramid.
        # so cor_planes = levels_of_pyr * flattened_lookup
        cor_planes = 4 * (2*3 + 1)**2 

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True))

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(4, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True))

        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip(),
            nn.Sigmoid())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip())

        self.gru = ConvGRU(128, 128+128+64)
        self.agg = GraphAgg()
    def forward(self, net, inp, corr, flow=None, ii=None, jj=None):
        """ RaftSLAM update operator 
        this forward of updatemodule is called just for one image"""

        batch, num, ch, ht, wd = net.shape  # num is here number of edges for which we are doing update.
        
        """Ques: How do we initializing depths and camera poses in this iterative optimization algorithm???
        We don't need to initialise depth and poses...This line is where first update operator is applied. 
        And if flow is None, we are initialising motion features i.e.,[flow, residual] to 0... 
        inturn we don't need to initialise initial poses and depths now for flow... 
        But we need to initialise pose and depths individually to do the 1st correlation lookup.
        (we've directly initialized motion[flow,residual])."""
        if flow is None:
            flow = torch.zeros(batch, num, 4, ht, wd, device=net.device)

        output_dim = (batch, num, -1, ht, wd)
        net = net.view(batch*num, -1, ht, wd)
        inp = inp.view(batch*num, -1, ht, wd)
        corr = corr.view(batch*num, -1, ht, wd)
        flow = flow.view(batch*num, -1, ht, wd)

        corr = self.corr_encoder(corr)
        """DEBUG: check the size of flow before and after: """
        flow = self.flow_encoder(flow)
        net = self.gru(net, inp, corr, flow)

        ### update variables ###
        #flow revisions
        delta = self.delta(net).view(*output_dim)
        # confidence map
        weight = self.weight(net).view(*output_dim)
        """DEBUG: check the size of delta before and after: """
        delta = delta.permute(0,1,3,4,2)[...,:2].contiguous()
        weight = weight.permute(0,1,3,4,2)[...,:2].contiguous()

        net = net.view(*output_dim)

        if ii is not None:
            """Took from Paper: 
            pool(scatter_mean) the hidden state over all features which share the same source view i and 
            predict a pixel-wise damping factor λ ("eta" here, and "damping" in other .py files, 
            and again "eta" in cuda code).
            Additionally, we use the pooled features to predict a 8x8 mask which can be used to 
            upsample the inverse depth estimate."""
            eta, upmask = self.agg(net, ii.to(net.device))
            return net, delta, weight, eta, upmask

        else:
            return net, delta, weight


class DroidNet(nn.Module):
    def __init__(self):
        super(DroidNet, self).__init__()
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        self.cnet = BasicEncoder(output_dim=256, norm_fn='none')
        self.update = UpdateModule()


    def extract_features(self, images):
        """ run feature extraction networks : to send to correlation module"""

        # normalize images
        images = images[:, :, [2,1,0]] / 255.0
        """Debug: check how many images are comming here: Ans: """

        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])

        fmaps = self.fnet(images)
        net = self.cnet(images)
        
        net, inp = net.split([128,128], dim=2)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        return fmaps, net, inp


    def forward(self, Gs, images, disps, intrinsics, graph=None, num_steps=12, fixedp=2):
        """ Estimates SE3 or Sim3 between pair of frames """

        # u = keyframe_indicies(graph)

        """DEBUG: what is the size of ii and jj... is it just one edge or list of edges ?? """
        ii, jj, kk = graph_to_edge_list(graph)
        """ ii has starting-nodes and jj has ending-nodes of the edges 
        ex: edge0 is from node ii[0] to jj[0] and so on.
            edge1 is from node ii[1] to jj[1] and so on.
        So, (ii,jj) will together represent the edges of the graph.
        """

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)

        """DEBUG: check the size of fmaps,net, inp: Is it for one image or for all images? Ans:  """
        fmaps, net, inp = self.extract_features(images)

        """DEBUG: try to print(ii) to see what values are coming here"""
        net, inp = net[:,ii], inp[:,ii]
        corr_fn = CorrBlock(fmaps[:,ii], fmaps[:,jj], num_levels=4, radius=3) 

        ht, wd = images.shape[-2:]
        coords0 = pops.coords_grid(ht//8, wd//8, device=images.device)

        """ 
        this function map points from frame_ii -> frame_jj 
        coords1 is dense correspondence field. [ which can be used to calc the induced flow by (induced_flow = coords1 - coords0)]
        """

        #In below line: ii is starting frame(or starting node of cov. graph) and jj is ending frame(or ending node of cov. graph)
        """ pops.projective_transform() map points from frame_ii->frame_jj  (ii to jj is an edge in covisibility graph)
        it'll find the correspondence-field (corresponding points of frame_ii in frame_jj )"""
        coords1, _ = pops.projective_transform(Gs, disps, intrinsics, ii, jj)   # for edge ii->jj (which represents a pair of frames)
        target = coords1.clone()

        Gs_list, disp_list, residual_list = [], [], []
        for step in range(num_steps):
            Gs = Gs.detach()
            disps = disps.detach()
            coords1 = coords1.detach()
            target = target.detach()

            #Correlation Lookup
            corr = corr_fn(coords1)
            
            # extract motion features
            #DCF is "Dense Correspondence Field" : which means corresponding pixels of frame_ii in frame_jj
            #below is residual = prev_pred_DCF - current_DCF..... (where prev_pred_DCF = prev_DCF + prev_delta)
            resd = target - coords1
            
            #below is induced flow(= DCF - coords0)
            flow = coords1 - coords0

            # concatenate motion features
            motion = torch.cat([flow, resd], dim=-1)
            motion = motion.permute(0,1,4,2,3).clamp(-64.0, 64.0)

            # Note that this update and all the steps in this function is for frame pair (ii,jj)
            net, delta, weight, eta, upmask = \
                self.update(net, inp, corr, motion, ii, jj)

            # Srj: update flow with flow-revision
            # target is predicted DCF (= current_DCF + current_delta)
            # target is one step away from predicted flow (predicted_fow = target - coords0)
            target = coords1 + delta

            for i in range(2):
                Gs, disps = BA(target, weight, eta, Gs, disps, intrinsics, ii, jj, fixedp=2) # fixedp means fix poses (i.e. fix first 2 poses)

            """DEBUG Training:
            All shapes that are passed in above function when BA is called for first set of 7 frames:

            weight.shape = torch.Size([1, 22, 48, 64, 2])
            target.shape = torch.Size([1, 22, 48, 64, 2])
            eta.shape = torch.Size([1, 7, 48, 64])
            Gs is a SE3 type object, and it is not a tensor
            Gs.shape = torch.Size([1, 7])   ..... Gs.tangent_shape = torch.Size([1, 7, 6])
            disps.shape = torch.Size([1, 7, 48, 64])
            intrinsics.shape = torch.Size([1, 7, 4])
            ii.shape = torch.Size([22])
            jj.shape = torch.Size([22])
            """

            # coords1 is DCF for next iteration. 
            coords1, valid_mask = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
            # Similarly residual is residual for next iteration.
            residual = (target - coords1)
            
            # we are appending Gs, disps and residual for each update of (convGRU update + BA) ... (#updates = num_steps = 15 while training)
            Gs_list.append(Gs)
            disp_list.append(upsample_disp(disps, upmask))
            residual_list.append(valid_mask * residual)


        return Gs_list, disp_list, residual_list
