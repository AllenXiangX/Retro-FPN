
import os
import torch.utils.data as data
import torch
import numpy as np
from os.path import join, exists
from glob import glob
import multiprocessing as mp
import SharedArray as SA
import dataset.augmentation as t
from dataset.voxelizer import Voxelizer


def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def collation_fn(batch):
    """
    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)

    """
    coords, feats, labels = list(zip(*batch))
    batch_n = []
    for i in range(len(coords)):
        coords[i][:, 0] *= i
        batch_n.append(coords[i].shape[0])

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), torch.IntTensor(batch_n)


def collation_fn_eval_all(batch):
    """
    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)

    """
    coords, feats, labels, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)
    batch_n = []

    accmulate_points_num = 0
    for i in range(len(coords)):
        coords[i][:, 0] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]
        batch_n.append(coords[i].shape[0])

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), torch.cat(inds_recons), torch.IntTensor(batch_n)


class S3DIS(data.Dataset):
    # Augmentation arguments
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2

    def __init__(self, dataPathPrefix='Data', voxelSize=0.05, test_area=5,
                 split='train', aug=False, memCacheInit=False, identifier=1233, loop=1,
                 data_aug_color_trans_ratio=0.1, data_aug_color_jitter_std=0.05, data_aug_hue_max=0.5,
                 data_aug_saturation_max=0.2, eval_all=False, voxel_max=None
                 ):
        super(S3DIS, self).__init__()
        self.split = split
        print('test area: ', test_area)
        data_root = dataPathPrefix
        data_list = sorted(os.listdir(dataPathPrefix))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]

        if split == 'train':
            self.data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
        else:
            self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item]


        
        for item in self.data_list:
            if not os.path.exists("/dev/shm/{}".format(item)):
                data_path = os.path.join(data_root, item + '.npy')
                data = np.load(data_path)  # xyzrgbl, N*7
                sa_create("shm://{}".format(item), data)

        self.data_idx = np.arange(len(self.data_list))
        self.voxelSize = voxelSize
        self.aug = aug
        self.loop = loop
        self.eval_all = eval_all
        self.voxel_max = voxel_max
        self.voxelizer = Voxelizer(
            voxel_size=voxelSize,
            clip_bound=None,
            use_augmentation=True,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)

        if aug:
            prevoxel_transform_train = [t.ElasticDistortion(self.ELASTIC_DISTORT_PARAMS)]
            self.prevoxel_transforms = t.Compose(prevoxel_transform_train)
            input_transforms = [
                t.ChromaticAutoContrast(),
                t.ChromaticTranslation(data_aug_color_trans_ratio),
                t.ChromaticJitter(data_aug_color_jitter_std),
                t.HueSaturationTranslation(data_aug_hue_max, data_aug_saturation_max)
            ]
            self.input_transforms = t.Compose(input_transforms)

        print('[*] %s (%s) loading done (%d)! ' % (dataPathPrefix, split, len(self.data_list)))

    def __getitem__(self, index_long):
        data_idx = self.data_idx[index_long % len(self.data_idx)]
        data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        locs_in, feats_in, labels_in = data[:, 0:3], data[:, 3:6], data[:, 6]

        locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in
        locs, feats, labels, inds_reconstruct = self.voxelizer.voxelize(locs, feats_in, labels_in)
        if self.eval_all:
            labels = labels_in
        if self.aug:
            locs, feats, labels = self.input_transforms(locs, feats, labels)

        coord_min = np.min(locs, axis=0)
        locs -= coord_min
        coords = torch.from_numpy(locs).int()
        coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
        feats = torch.from_numpy(feats).float() / 127.5 - 1.
        labels = torch.from_numpy(labels).long()

        if self.eval_all:
            return coords, feats, labels, torch.from_numpy(inds_reconstruct).long()

        if self.voxel_max and coords.shape[0] > self.voxel_max:
            init_idx = np.random.randint(labels.shape[0]) if 'train' in self.split else labels.shape[0] // 2
            crop_idx = torch.argsort(torch.sum(torch.square(coords - coords[init_idx]), 1))[:self.voxel_max]
            coords, feats, labels = coords[crop_idx], feats[crop_idx], labels[crop_idx]

        return coords, feats, labels

    def __len__(self):
        return len(self.data_list) * self.loop