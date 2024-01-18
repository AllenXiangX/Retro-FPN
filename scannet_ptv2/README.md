## Installation

### Requirements

- Ubuntu: 18.04 or higher

- CUDA: 10.2 or higher

- PyTorch: 1.10.0 ~ 1.11.0

- Hardware: 4 x 24G memory GPUs or better

### Conda Environment

```bash
conda create -n pcr python=3.8 -y
conda activate pcr
conda install ninja -y
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y
conda install -c anaconda h5py pyyaml -y
conda install -c conda-forge sharedarray tensorboardx yapf addict einops scipy plyfile termcolor timm -y
conda install -c pyg pytorch-cluster pytorch-scatter pytorch-sparse -y
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu113
```

### Optional Installation

```bash
# Open3D (Visualization)
pip install open3d

# PTv1 & PTv2
cd libs/pointops
python setup.py install
cd ../..

# stratified transformer
pip install torch-points3d

# fix dependence, caused by install torch-points3d 
pip uninstall SharedArray
pip install SharedArray==3.2.1

cd libs/pointops2
python setup.py install
cd ../..

# MinkowskiEngine (SparseUNet)
# refer https://github.com/NVIDIA/MinkowskiEngine

# torchsparse (SPVCNN)
# refer https://github.com/mit-han-lab/torchsparse
# install method without sudo apt install
conda install google-sparsehash -c bioconda
export C_INCLUDE_PATH=${CONDA_PREFIX}/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=${CONDA_PREFIX}/include:CPLUS_INCLUDE_PATH
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git
```

## Data Preparation

Please refer to [PointTransformer V2](https://github.com/Pointcept/PointTransformerV2) for preparing data.

## Quick Start

### Training

**Train from scratch.** The training processing is based on configs in `configs` folder. 
The training script will generate an experiment folder in `exp` folder and backup essential code in the experiment folder.
Training config, log, tensorboard and checkpoints will also be saved into the experiment folder during the training process.

```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME}
```

For example:

```bash
sh scripts/train.sh -p python -d scannet -c semseg-retrofpn -n semseg-retrofpn
```

### Testing

```bash
sh scripts/test.sh -p ${INTERPRETER_PATH} -d ${DATASET_NAME} -n ${EXP_NAME} -w ${CHECKPOINT_NAME}
```

For example:

```bash
sh scripts/test.sh -p python -d scannet -n semseg-retrofpn -w model_best
```

## Pre-trained Models

We provide the pretrained models for the Point Transformer V2 +  Retro-FPN on Scannet. Please download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/1wmtZIGlMFYZQ_zY7ghgNDFyB_PDXDa1P?usp=sharing) .

### Acknowledgement
This repo is based on [Point Transformer V2](https://github.com/Pointcept/PointTransformerV2). We thank the authors for their great job!


## References

If you use this code, please cite:
```
@InProceedings{xp_retrofpn_2023,
    author    = {Xiang, Peng and Wen, Xin and Liu, Yu-Shen and Zhang, Hui and Fang, Yi and Han, Zhizhong},
    title     = {Retro-FPN: Retrospective Feature Pyramid Network for Point Cloud Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {17826-17838}
}


@inproceedings{zhao2021point,
  title={Point transformer},
  author={Zhao, Hengshuang and Jiang, Li and Jia, Jiaya and Torr, Philip HS and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16259--16268},
  year={2021}
}
```
