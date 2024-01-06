# Installation history for DROID-SLAM in Vision04 server (A100 GPUs)

droidenv5 -> totally their's
droidenv -> clone of base(have pytorch,etc) and then envirionment2.yaml (deleted)
droidenv2 -> clone of base(have pytorch,etc) and then envirionment.yaml

droidenv3 -> environment3.yaml

droidenv4 -> environment5.yaml
droidenv0 -> environment0.yaml

*now doing in scratch/DROID-SLAM*
droidenv -> environment_novis.yaml ( with 2+2 lines(for sm_80 and sm_86) commented in setup.py) 

droid_novis -> environment_novis2.yaml ( with 2 lines(for sm_86) commented in setup.py) -> it rav v. well, no errors in installing setup.py, just few warnings. 

## Final Working envs

- `droid_novis2` -> environment_novis3.yaml ( with 2 lines(for sm_86) commented in setup.py) -> it ran v. well, no errors in installing setup.py, just few warnings.
- **Pytorch was not able to compile using cuda, so Finally I uninstalled torch and ran below command to install it using pip**

- `droid_vis` -> cloned droid_novis2, then done `pip install open3d` 
  - output of `~/check_cuda.py` for droid_vis is:
  ```
    torch.__version__= 1.9.0+cu111

    torch.version.cuda =  11.1

    torch.cuda.is_available() =  True

    torch.cuda.get_arch_list() =  ['sm_37', 'sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86']

    Architecture of vision server is sm_80

    torch.Tensor([10,23]).to("cuda") = tensor([10., 23.], device='cuda:0')
  ```






