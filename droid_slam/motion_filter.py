import cv2
import torch
import lietorch

from collections import OrderedDict
from droid_net import DroidNet

import geom.projective_ops as pops
from modules.corr import CorrBlock


class MotionFilter:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self, net, video, thresh=2.5, device="cuda:0"):
        
        """DEBUG: what is net here:"""
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth=None, intrinsics=None, gt_pose = None, gt_depth = None):
        """ main update operation - run on every frame in video """

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8

        # normalize images
        inputs = image[None, :, [2,1,0]].to(self.device) / 255.0 # [2,1,0] for BGR->RGB conversion
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        """NEW FRAME .... gmap is new frame, and fmap is previous frame."""
        gmap = self.__feature_encoder(inputs)

        ### always add first frame to the depth video ###
        # So no need to check for any motion for 1st frame.... Just add it.
        if self.video.counter.value == 0:
            #initialise hidden-state(net) and input(inp) using the first frame.
            net, inp = self.__context_encoder(inputs[:,[0]])
            self.net, self.inp, self.fmap = net, inp, gmap #storing gmap in fmap for next iteration. So fmap is always the previous frame.
            self.video.append(tstamp, image[0], Id, 1.0, depth, intrinsics / 8.0, gmap, net[0,0], inp[0,0], gt_pose, gt_depth)

        ### only add new frame if there is enough motion b/w current frame(gmap) and previous frame(fmap).###
        else:                
            # index for correlation volume [ it just creates a meshgrid.]
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]

            
            """
            In below line we are doing correlation lookup from the whole correlation pyramid.
            they have CALLED that object also for coords0 only(instead for coords1 which happens at droid_net.py)... 
            so it will call the __call__() function of Corrblock after making the object.
            fmap is PREVIOUS frame, and gmap is NEW frame.
            Note that we are not finding any correspondence-field here for coords0, we are just taking correspondence-field as coords0.
            """
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, _ = self.update(self.net[None], self.inp[None], corr)

            # check motion magnitue / add new frame to video
            if delta.norm(dim=-1).mean().item() > self.thresh:
                self.count = 0
                """ 
                by suraj: we are passing to context encoder only if there is enough motion.
                and in that case also saving the context encoder output(self.net, self.inp, self.fmap) 
                using self.video.append() function
                that's why they have made different function for context-encoder and feature-encoder in the above code.(not used net.extract_features() as it computes both)
                """
                
                net, inp = self.__context_encoder(inputs[:,[0]])
                self.net, self.inp, self.fmap = net, inp, gmap  #storing gmap in fmap for next iteration. So fmap is always the last frame processed(only if it had enough motion).
                # add new frame to video and increase the counter.value of video by 1.
                self.video.append(tstamp, image[0], None, None, depth, intrinsics / 8.0, gmap, net[0], inp[0], gt_pose, gt_depth)

            else:
                # not enough motion, then totally skip(ignore) that frame (and fmap is not updated and same as last processed frame)
                #Srj: it is counting the number of consecutive frames that are not added to the video.
                self.count += 1




# class MotionFilter:
#     """ This class is used to filter incoming frames and extract features """

#     def __init__(self, net, video, thresh=2.5, device="cuda:0"):
        
#         # split net modules
#         self.cnet = net.cnet
#         self.fnet = net.fnet
#         self.update = net.update

#         self.video = video
#         self.thresh = thresh
#         self.device = device

#         self.count = 0

#         # mean, std for image normalization
#         self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
#         self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
#     @torch.cuda.amp.autocast(enabled=True)
#     def __context_encoder(self, image):
#         """ context features """
#         x = self.cnet(image)
#         net, inp = self.cnet(image).split([128,128], dim=2)
#         return net.tanh().squeeze(0), inp.relu().squeeze(0)

#     @torch.cuda.amp.autocast(enabled=True)
#     def __feature_encoder(self, image):
#         """ features for correlation volume """
#         return self.fnet(image).squeeze(0)

#     @torch.cuda.amp.autocast(enabled=True)
#     @torch.no_grad()
#     def track(self, tstamp, image, depth=None, intrinsics=None):
#         """ main update operation - run on every frame in video """

#         Id = lietorch.SE3.Identity(1,).data.squeeze()
#         ht = image.shape[-2] // 8
#         wd = image.shape[-1] // 8

#         # normalize images
#         inputs = image[None, None, [2,1,0]].to(self.device) / 255.0
#         inputs = inputs.sub_(self.MEAN).div_(self.STDV)

#         # extract features
#         gmap = self.__feature_encoder(inputs)

#         ### always add first frame to the depth video ###
#         if self.video.counter.value == 0:
#             net, inp = self.__context_encoder(inputs)
#             self.net, self.inp, self.fmap = net, inp, gmap
#             self.video.append(tstamp, image, Id, 1.0, intrinsics / 8.0, gmap[0], net[0], inp[0])

#         ### only add new frame if there is enough motion ###
#         else:                
#             # index correlation volume
#             coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
#             corr = CorrBlock(self.fmap[None], gmap[None])(coords0)

#             # approximate flow magnitude using 1 update iteration
#             _, delta, weight = self.update(self.net[None], self.inp[None], corr)

#             # check motion magnitue / add new frame to video
#             if delta.norm(dim=-1).mean().item() > self.thresh:
#                 self.count = 0
#                 net, inp = self.__context_encoder(inputs)
#                 self.net, self.inp, self.fmap = net, inp, gmap
#                 self.video.append(tstamp, image, None, None, intrinsics / 8.0, gmap[0], net[0], inp[0])

#             else:
#                 self.count += 1

