import lietorch
import torch
import torch.nn.functional as F

from .chol import block_solve, schur_solve
import geom.projective_ops as pops

from torch_scatter import scatter_sum

# utility functions for scattering ops
def safe_scatter_add_mat(A, ii, jj, n, m):
    # safe bcoz: to remove -ve node numbers[for deleted nodes(first 2 nodes)]. as ii and jj had negative values.
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m) # v is a boolean mask to remove negative node numbers
    # A[:,v].shape = torch.Size([1, 18, 6, 6])
    # A.shape = torch.Size([1, 24, 6, 6])
    # m = n = P = (7-2) = 5 ... So, dim_size =  m*n = 25
    # ii.shape = torch.Size([24]) ... ii[v].shape = torch.Size([18]) (same for jj[v])
    return scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m) # = torch.Size([1, 25, 6, 6])... 25=m*n
    #can't understand why they have done ii[v]*m + jj[v] ???
    
def safe_scatter_add_vec(b, ii, n):
    # safe bcoz: to remove -ve node numbers[for deleted nodes(first 2 nodes)]. as ii and jj had negative values.
    v = (ii >= 0) & (ii < n) # v is a boolean mask to remove negative node numbers
    # b.shape = torch.Size([1, 24, 6])... b[:,v].shape = torch.Size([1, 18, 6])
    return scatter_sum(b[:,v], ii[v], dim=1, dim_size=n) #it's shape is torch.Size([1, 5, 6])


# apply retraction operator to inv-depth maps
def disp_retr(disps, dz, ii):
    ii = ii.to(device=dz.device)
    return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])


# apply retraction operator to poses
def pose_retr(poses, dx, ii):
    #poses.data.shape = torch.Size([1, 7, 7])
    #dx.shape = torch.Size([1, 5, 6])
    #ii.shape = torch.Size([5]) ... ii = tensor([2, 3, 4, 5, 6])
    # Reason to use scatter_sum operation for pose_retr() is : we got poses update for each edges, 
    # but at the end we need to sum all those which belongs to same camera-frame.
    ii = ii.to(device=dx.device)
    return poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))

def BA(target, weight, eta, poses, disps, intrinsics, ii, jj, fixedp=1, rig=1):
    """ Full Bundle Adjustment """
    """ INPUT: predicted_flow, confidence_of_predicted_flow, eta, intrinsics, graph(ii,jj).
        OUTPUT: refined_pose, refined_disps 
    
    target.shape: torch.Size([1, 24, 48, 64, 2])
    weight.shape: torch.Size([1, 24, 48, 64, 2])
    eta.shape: torch.Size([1, 7, 48, 64])  
    poses.data.shape:  torch.Size([1, 7, 7]) (poses is SE3 object) (7 is #keyframes and the other is the #params reqd for 1 pose) (shape:torch.Size([1, 7]))
    disps.shape: torch.Size([1, 7, 48, 64]) ... intrinsics.shape: torch.Size([1, 7, 4]) ... ii.shape: torch.Size([24]) ...  jj.shape: torch.Size([24]) 

    So, the current state of factor graph is:              [  7 nodes(keyframes) AND 24 edges     ]."""

    B, P, ht, wd = disps.shape  # P is #keyframes
    N = ii.shape[0] # N is #edges
    D = poses.manifold_dim # D = 6

    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(poses, disps, intrinsics, ii, jj, jacobian=True)  
    # Ji.shape =torch.Size([1, edges, h, w, 2, 6]) , Jj.shape = torch.Size([1, 22, 48, 64, 2, 6]), 
    # Jz.shape =torch.Size([1, edges, h, w, 2, 1])
    # below is  residual = predicted_flow - induced_flow
    r = (target - coords).view(B, N, -1, 1) # r.shape = torch.Size([1, 24, 6144, 1]) ... 6144 = 48*64*2

    # valid: exclude points that are too close to camera(0 or 1). weight is confidence of predicted flow(weight.shape = torch.Size([1, 24, 48, 64, 2]))
    # w will have 0 confidence for all the invalid points(too close to camera) 
    # WHAT do you mean by POINTS? all the pixels of image at 1/8th resolution because we have backprojected each pixel of image-i to 3D then projected to image-j.
    w = .001 * (valid * weight).view(B, N, -1, 1)  # new confidence_map # valid.shape = torch.Size([1, 24, 48, 64, 1])

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D) # Ji.shape=torch.Size([1, 24, 6144, 6]) ...2HW = 6144 
    Jj = Jj.reshape(B, N, -1, D) # Jj.shape=torch.Size([1, 24, 6144, 6])

    wJiT = (w * Ji).transpose(2,3)  # elementwise multiplication
    wJjT = (w * Jj).transpose(2,3)  # elementwise multiplication

    Jz = Jz.reshape(B, N, ht*wd, -1)

    Hii = torch.matmul(wJiT, Ji) #torch.Size([1, 24, 6, 6])
    Hij = torch.matmul(wJiT, Jj) #torch.Size([1, 24, 6, 6])
    Hji = torch.matmul(wJjT, Ji) #torch.Size([1, 24, 6, 6])
    Hjj = torch.matmul(wJjT, Jj) #torch.Size([1, 24, 6, 6])

    vi = torch.matmul(wJiT, r).squeeze(-1) 
    vj = torch.matmul(wJjT, r).squeeze(-1)

    Ei = (wJiT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)
    Ej = (wJjT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)

    w = w.view(B, N, ht*wd, -1)
    r = r.view(B, N, ht*wd, -1)
    wk = torch.sum(w*r*Jz, dim=-1)
    Ck = torch.sum(w*Jz*Jz, dim=-1)

    kx, kk = torch.unique(ii, return_inverse=True) #shape of ii and kk is torch.Size([24]) and kx is torch.Size([7])
    M = kx.shape[0] # = torch.Size([7])...kx has all the unique nodes numbers (from whic edges are starting)

    # only optimize keyframe poses ... # P is #keyframes ... rig=1 (but WHat is rig??? let's ignore for now.)
    P = P // rig - fixedp # no. of actual keyframes poses to optimize ... P = 7-2 = 5
    #to remove warning ii = ii // rig - fixedp # subtract 2 from each keyframe index as we have removed first 2 keyframes(fixed).
    #to remove warning jj = jj // rig - fixedp # subtract 2 from each keyframe index as we have removed first 2 keyframes(fixed).
    ii = torch.div(ii, rig, rounding_mode='trunc') - fixedp # subtract 2 from each keyframe index as we have removed first 2 keyframes(fixed).
    jj = torch.div(jj, rig, rounding_mode='trunc') - fixedp # subtract 2 from each keyframe index as we have removed first 2 keyframes(fixed).
    
    # So by subtracting ii and jj by fixedp, Now 3rd keyframe(ind=2) will have index as 0 now.

    # B of schur complement
    # Hii.shape = torch.Size([1, 24, 6, 6]) ...  H.shape = torch.Size([1, 25, 6, 6])   ....   25 for 5*5
    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)
    
    

    # E of schur complement
    # Ei.shape = torch.Size([1, 24, 6, 3072]) ... E.shape = torch.Size([1, 35, 6, 3072])  ..... 35 for 5*7 (ii has 5 nodes(only those we need to update), kk has 7)
    E = safe_scatter_add_mat(Ei, ii, kk, P, M) + \
        safe_scatter_add_mat(Ej, jj, kk, P, M)
    
    # v of schur complement
    #vi.shape = torch.Size([1, 24, 6]) ... v.shape = torch.Size([1, 5, 6])
    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)
    
    # C and w of schur complement
    C = safe_scatter_add_vec(Ck, kk, M)  #Ck.shape = torch.Size([1, 24, 3072])  ...  C.shape = torch.Size([1, 7, 3072])
    w = safe_scatter_add_vec(wk, kk, M) #wk.shape = torch.Size([1, 24, 3072])  ...  w.shape = torch.Size([1, 7, 3072])

    C = C + eta.view(*C.shape) + 1e-7

    H = H.view(B, P, P, D, D)
    E = E.view(B, P, M, D, ht*wd)

    ### 3: solve the system ###
    # dx and dz are the updates to camera poses and depths respectively.
    dx, dz = schur_solve(H, E, C, v, w)
    
    ### 4: apply retraction ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)

    disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.0)

    return poses, disps


def MoBA(target, weight, eta, poses, disps, intrinsics, ii, jj, fixedp=1, rig=1):
    """ Motion only bundle adjustment """
    """FROM Paper: 
    We only perform full bundle adjustment on keyframe images.
    In order to recover the poses of non-keyframes, we perform motion-only bundle adjustment
    """
    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim

    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)

    r = (target - coords).view(B, N, -1, 1)
    w = .001 * (valid * weight).view(B, N, -1, 1)

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D)
    Jj = Jj.reshape(B, N, -1, D)
    wJiT = (w * Ji).transpose(2,3)
    wJjT = (w * Jj).transpose(2,3)

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    # only optimize keyframe poses
    P = P // rig - fixedp
    ii = ii // rig - fixedp
    jj = jj // rig - fixedp

    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)
    
    H = H.view(B, P, P, D, D)

    ### 3: solve the system ###
    dx = block_solve(H, v)

    ### 4: apply retraction ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    return poses

