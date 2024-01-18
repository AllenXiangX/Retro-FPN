"""
Dataset Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""


from pcr.utils.registry import Registry

DATASETS = Registry('datasets')


def build_dataset(cfg):
    """Build test_datasets."""
    return DATASETS.build(cfg)
