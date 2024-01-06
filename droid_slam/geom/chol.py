import torch
import torch.nn.functional as F
import geom.projective_ops as pops

class CholeskySolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        # don't crash training if cholesky decomp fails
        try:
            # cholesky decomposition (get the Lower Triangular matrix) (as we know H = LL^T )
            U = torch.linalg.cholesky(H)
            xs = torch.cholesky_solve(b, U)
            ctx.save_for_backward(U, xs)
            ctx.failed = False
        except Exception as e:
            print(e)
            ctx.failed = True
            xs = torch.zeros_like(b)

        return xs

    #srj gradient is passing through this, but Is there any learnable parameters here? NO.
    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None

        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1,-2))

        return dH, dz

def block_solve(H, b, ep=0.1, lm=0.0001):
    """ solve normal equations """
    B, N, _, D, _ = H.shape
    I = torch.eye(D).to(H.device)
    H = H + (ep + lm*H) * I

    H = H.permute(0,1,3,2,4)
    H = H.reshape(B, N*D, N*D)
    b = b.reshape(B, N*D, 1)

    x = CholeskySolver.apply(H,b)
    return x.reshape(B, N, D)


def schur_solve(H, E, C, v, w, ep=0.1, lm=0.0001, sless=False):
    """ solve using shur complement """
    """understood completely"""
    """given :all the components(block matrices and vectors) of Equation 6 of "BA in the Large" paper...
            here H is B in my equations"""

    """output: 
    it is used to get the 
    - delta_y(update to camera poses) (here dx) and 
    - delta_z(update to 3d points) (but here it is update to depths)  (here dz)
    So, dx and dz are the updates to camera poses and depths respectively.
    """

    B, P, M, D, HW = E.shape  # E.shape = torch.Size([1, 5, 7, 6, 3072])
    # P: number of camera poses to update, 
    # D: number of parameters per pose,     .... So, P*D would be the total # camera parameters to update
    # M: number of depthmaps to update, 
    # HW: number of pixels per depthmap      .... So, M*HW would be the total # depth parameters to update

    H = H.permute(0,1,3,2,4).reshape(B, P*D, P*D)
    E = E.permute(0,1,3,2,4).reshape(B, P*D, M*HW)
    # Q is the inverse of C
    Q = (1.0 / C).view(B, M*HW, 1)

    #orig comment:  damping
    I = torch.eye(P*D).to(H.device)  # = the size of H ... PD would be the total # camera parameters to update
    # added ep and lm*H to diagonal of H [damping]
    H = H + (ep + lm*H) * I
    
    v = v.reshape(B, P*D, 1)
    w = w.reshape(B, M*HW, 1)

    Et = E.transpose(1,2)

    # S is schur complement of C in the Block matrix. Notation of below H is B in my equations(BA paper)
    S = H - torch.matmul(E, Q*Et)  # E.shape = torch.Size([1, 30, 21504]) and Q.shape = torch.Size([1, 21504, 1]) and Et.shape = torch.Size([1, 21504, 30])
    v = v - torch.matmul(E, Q*w)

    # takes matrix to be inverted S and vector b (v here)
    dx = CholeskySolver.apply(S, v) # here dx.shape = torch.Size([1, 30, 1])
    if sless:
        return dx.reshape(B, P, D)

    dz = Q * (w - Et @ dx)  # here dz.shape = torch.Size([1, 21504, 1])  ... 21504 = 7*3072 = 7*48*64
    #reshape the delta vectors.
    dx = dx.reshape(B, P, D)
    dz = dz.reshape(B, M, HW)

    # dx and dz are the updates to camera poses and depths respectively.
    """DEBUG Training: check the size of dx an dz to confirm above comment.  (done)
    dx = torch.Size([1, 5, 6])  [5 camera poses(as first 2 camera poses are fixed to remove the gauge freedom) and 6 parameters per pose]
    dz = torch.Size([1, 7, 3072]) on resizing it'll become torch.Size([1, 7, 48, 64]) 
    """
    return dx, dz