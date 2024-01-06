
## DROID-SLAM installation guide.

Made a working env for DROID-SLAM from scratch in vision04(A100 GPU) using these commands on 6Jan2024... So it works... env name is `droid`
Make a new conda environment, install these in given order: 
- `conda create -n droid python=3.9`

- for pytorch,etc: `pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`

- `pip install open3d`

- `pip install opencv-python matplotlib tqdm pyyaml`

**To install lietorch and droid_backends:** moreover I had also added the print commands in the droid_kernels.cu (line number 1424... function ba_cuda()) to debug
- `python setup.py install` (for installing in A100 GPUS: comment out the last line of the setup.py that was installing for 'sm_86' aarchitecture... A100 is based on sm_80, but it also supports all earlier ones.)
    - if you want to make any changes in the ba_cuda() code, then after making those changes in droid_kernels.cu, you need to do `python setup.py install` from the conda env in which you want to install the changed version. 
    **Note:** it doesn't depend on which repo(you can have DROID-SLAM code at multiple locations). It just depend on from which conda env you run the program. So the installed libraries after `python setup.py install` goes to the conda env path.

- `pip install torch-scatter==2.0.9` (as this version is also in my earleir env: `droid_novis2`)

