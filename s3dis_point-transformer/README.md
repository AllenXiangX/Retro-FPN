## The integration of Retro-FPN with Point Transformer

### Setup environment

```bash
conda create -n pt python=3.7
conda activate pt
conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia -y
pip install -r requirements.txt

cd lib/pointops
python3 setup.py install
cd ../..
```

### Data preparation
Download [S3DIS](https://drive.google.com/uc?export=download&id=1KUxWagmEWnvMhEb4FRwq2Mj0aa3U3xUf) dataset and symlink the paths to them as follows:
```bash
mkdir -p data
ln -s /path_to_s3dis_dataset dataset/s3dis
```

### Usage
- Train
  - specify the configurations on the [config file](./config) and do:
  ```
  python train.py --config=config/s3dis/s3dis_retrofpn.yaml
  ```

- Test
  - specify the configurations on the [config file](./config) and do:
  ```
  python test.py --config=config/s3dis/s3dis_retrofpn.yaml
  ```

### Acknowledgement
This repo is based on [Point Transformer](https://github.com/POSTECH-CVLab/point-transformer). We thank the authors for their great job!


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


@inproceedings{zhao2021point,
  title={Point transformer},
  author={Zhao, Hengshuang and Jiang, Li and Jia, Jiaya and Torr, Philip HS and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16259--16268},
  year={2021}
}
```