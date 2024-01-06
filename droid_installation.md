
## DROID-SLAM installation guide.

Made a working env for DROID-SLAM from scratch in vision04(A100 GPU) using these commands on 6Jan2024... So it works... env name is `droid`
Make a new conda environment, install these in given order: 
- `conda create -n droid python=3.9`

- for pytorch,etc: `pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`

- `pip install open3d`

- `pip install opencv-python matplotlib tqdm pyyaml`

**To install lietorch and droid_backends:** moreover I had also added the print commands in the droid_kernels.cu (line number 1424... function ba_cuda()) to debug
- `python setup.py install` (for installing in A100 GPUS: comment out the last line of the setup.py that was installing for 'sm_86' aarchitecture... A100 is based on sm_80, but it also supports all earlier ones.)

- `pip install torch-scatter==2.0.9` (as this version is also in my earleir env: `droid_novis2`)

