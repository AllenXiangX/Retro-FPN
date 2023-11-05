## The integration of Retro-FPN with MinkowskiNet

### Setup environment

```bash
conda create -n mink python=3.7
conda activate mink
conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia -y
pip install -r requirements.txt

cd lib/pointops
python3 setup.py install
cd ../..

cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

### Data preparation
- S3DIS

  Download [S3DIS](https://drive.google.com/uc?export=download&id=1KUxWagmEWnvMhEb4FRwq2Mj0aa3U3xUf) dataset and symlink the paths to them as follows:
  ```bash
  mkdir -p data
  ln -s /path_to_s3dis_dataset data/s3dis
  ``` 

- ScanNet
  - Download the dataset from official website.
  - Use dataset/preprocess_3d_scannet.py to process the data.
  - symlink the paths to them as follows:
  ```bash
  mkdir -p data
  ln -s /path_to_scannet_dataset data/ScanNet
  ``` 

### Pre-trained model
We provide the following pretrained models:

| Dataset |  Train |  Model |
|----|----|----|
| S3DIS | [log](https://drive.google.com/file/d/12nuOQK3CXod9-wsnhwSWZCe-u3RGzfdF/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1jzlAKVJCtlTYLqhg7adnv8JkHZyojx9W/view?usp=drive_link) |
| ScanNet | [log](https://drive.google.com/file/d/1Z9jwsl14HMkTbiuZuwBlhN4Jptqra5TW/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1HAOxlbNm2aFkTEIiVRJhBQYCWda1ZBaR/view?usp=drive_link) |

### Usage
- Train
  - specify the configurations on the [config file](./config) and run:
  ```
  python train.py --config=path_to_configuration_file
  ```

- Test
  - specify the configurations on the [config file](./config) and run:
  ```
  python test.py --config=path_to_configuration_file
  ```

### Acknowledgement
This repo is based on [BPNet](https://github.com/wbhu/BPNet), [Point Transformer](https://github.com/POSTECH-CVLab/point-transformer), and [MinkowskiNet](https://github.com/NVIDIA/MinkowskiEngine/tree/master). We thank the authors for their great job!


### References

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

@inproceedings{hu-2021-bidirectional,
        author      = {Wenbo Hu, Hengshuang Zhao, Li Jiang, Jiaya Jia and Tien-Tsin Wong},
        title       = {Bidirectional Projection Network for Cross Dimensional Scene Understanding},
        booktitle   = {CVPR},
        year        = {2021}
    }

@inproceedings{choy20194d,
  title={4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks},
  author={Choy, Christopher and Gwak, JunYoung and Savarese, Silvio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3075--3084},
  year={2019}
}


@inproceedings{zhao2021point,
  title={Point transformer},
  author={Zhao, Hengshuang and Jiang, Li and Jia, Jiaya and Torr, Philip HS and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16259--16268},
  year={2021}
}
```