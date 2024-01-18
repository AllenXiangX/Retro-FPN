# Retro-FPN: Retrospective Feature Pyramid Network for Point Cloud Semantic Segmentation (ICCV 2023)

[Peng Xiang*](https://scholar.google.com/citations?user=Bp-ceOAAAAAJ&hl=zh-CN&oi=sra), [Xin Wen*](https://scholar.google.com/citations?user=7gcGzs8AAAAJ&hl=zh-CN&oi=sra), [Yu-Shen Liu](http://cgcad.thss.tsinghua.edu.cn/liuyushen/), [Hui Zhang](https://www.thss.tsinghua.edu.cn/en/faculty/huizhang.htm), [Yi Fang](https://scholar.google.com/citations?user=j-cyhzwAAAAJ&hl=en), [Zhizhong Han](https://h312h.github.io/)

[<img src="./pics/retrofpn.jpg" width="100%" alt="Intro pic" />](pics/retrofpn.jpg)

## [Retro-FPN]

**Retro-FPN: Retrospective Feature Pyramid Network for Point Cloud Semantic Segmentation**

> Learning per-point semantic features from the hierarchical feature pyramid is essential for point cloud semantic segmentation. However, most previous methods suffered from ambiguous region features or failed to refine per-point features effectively, which leads to information loss and ambiguous semantic identification. To resolve this, we propose Retro-FPN to model the per-point feature prediction as an explicit and retrospective refining process, which goes through all the pyramid layers to extract semantic features explicitly for each point. Its key novelty is a retro-transformer for summarizing semantic contexts from the previous layer and accordingly refining the features in the current stage. In this way, the categorization of each point is conditioned on its local semantic pattern. Specifically, the retro-transformer consists of a local cross-attention block and a semantic gate unit. The cross-attention serves to summarize the semantic pattern retrospectively from the previous layer. And the gate unit carefully incorporates the summarized contexts and refines the current semantic features. Retro-FPN is a pluggable neural network that applies to hierarchical decoders. By integrating Retro-FPN with three representative backbones, including both point-based and voxel-based methods, we show that Retro-FPN can significantly improve performance over state-of-the-art backbones. Comprehensive experiments on widely used benchmarks can justify the effectiveness of our design.



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
Please refer to [PointTransformer V2](https://github.com/Pointcept/PointTransformerV2) for preparaing data

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
cd scannet
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

## [Cite this work]

```
@InProceedings{xp_retrofpn_2023,
    author    = {Xiang, Peng and Wen, Xin and Liu, Yu-Shen and Zhang, Hui and Fang, Yi and Han, Zhizhong},
    title     = {Retro-FPN: Retrospective Feature Pyramid Network for Point Cloud Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {17826-17838}
}

```

## Acknowledgements

Some of the code of this repo is borrowed from: 
- [Point Transformer](https://github.com/POSTECH-CVLab/point-transformer)
- [Point Transformer V2](https://github.com/POSTECH-CVLab/point-transformer)
- [BPNet](https://github.com/wbhu/BPNet)
- [MinkowskiNet](https://github.com/NVIDIA/MinkowskiEngine/tree/master)


We thank the authors for their great job!

## License

This project is open sourced under MIT license.
