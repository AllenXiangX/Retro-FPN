"""
Model Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pcr.utils.registry import Registry

MODELS = Registry('models')
MODULES = Registry('modules')


def build_model(cfg):
    """Build test_datasets."""
    return MODELS.build(cfg)
