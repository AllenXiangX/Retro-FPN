"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


from pcr.engines.defaults import default_argument_parser, default_config_parser, default_setup, Trainer
from pcr.engines.launch import launch
import os


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = Trainer(cfg)
    trainer.train()


def main():

    args = default_argument_parser().parse_args() # origin
    # args.config_file = '/localdata_ssd/xiaoyu/xiangp/code/ptv2/configs/scannet/semseg-retrov2m2-0-base.py'
    # args.options = {'save_path': '../exp/scannet/48_head_12k_4layers_1111'}
    # args.num_gpus = 1


    # print(args.config_file)
    # print(args.options)
    # exit(0)

    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
