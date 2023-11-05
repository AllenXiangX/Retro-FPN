# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
                'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
                'otherfurniture']
UNKNOWN_ID = 20
N_CLASSES = len(CLASS_LABELS)


CLASS_LABELS_S3DIS = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'chair',
                 'table','bookcase','sofa','board','clutter',]
N_CLASSES_S3DIS = len(CLASS_LABELS_S3DIS)


def confusion_matrix(pred_ids, gt_ids):
    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs = gt_ids != UNKNOWN_ID
    return np.bincount(pred_ids[idxs] * 20 + gt_ids[idxs], minlength=400).reshape((20, 20)).astype(np.ulonglong)


def get_iou(label_id, confusion):
    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return float(tp) / denom, tp, denom


def evaluate(pred_ids, gt_ids, stdout=False, data_name='scannet'):
    if stdout:
        print('evaluating', gt_ids.size, 'points...')
    confusion = confusion_matrix(pred_ids, gt_ids)
    class_ious = {}
    mean_iou = 0

    if data_name == 'scannet':
        n_classes = N_CLASSES
        class_labels = CLASS_LABELS
    elif data_name == 's3dis':
        n_classes = N_CLASSES_S3DIS
        class_labels = CLASS_LABELS_S3DIS
    else:
        raise Exception('wrong dataset name')
    
    for i in range(n_classes):
        label_name = class_labels[i]
        class_ious[label_name] = get_iou(i, confusion)
        mean_iou += class_ious[label_name][0] / n_classes

    if stdout:
        print('classes          IoU')
        print('----------------------------')
        for i in range(n_classes):
            label_name = class_labels[i]
            print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, class_ious[label_name][0],
                                                                   class_ious[label_name][1],
                                                                   class_ious[label_name][2]))
        print('mean IOU', mean_iou)
    return mean_iou
