import os
import time
import random
import numpy as np
import logging
import argparse
os.environ["OMP_NUM_THREADS"] = '16'
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

from MinkowskiEngine import SparseTensor, CoordsManager
from util import config
from util.util import AverageMeter, intersectionAndUnionGPU, poly_learning_rate, save_checkpoint


best_iou = 0.0


def worker_init_fn(worker_id):
    random.seed(time.time() + worker_id)


def get_parser():
    parser = argparse.ArgumentParser(description='RetroFPN')
    parser.add_argument('--config', type=str, default='config/s3dis/retro_5cm.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/retro_5cm.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger(args):
    logger_name = "main-logger"
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=os.path.join(args.save_path, "train.log"))
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    handler2.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.addHandler(handler2)
    return logger


def main_process():
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        # cudnn.benchmark = False
        # cudnn.deterministic = True
    # Log for check version
    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    if args.data_name == 'scannet=':
        from dataset.scanNet3D import ScanNet3D, collation_fn
        _ = ScanNet3D(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split='train', aug=args.aug,
                      memCacheInit=True, loop=5)
        if args.evaluate:
            _ = ScanNet3D(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split='val', aug=args.aug,
                          memCacheInit=True)
    elif args.data_name == 's3dis':
        from dataset.S3DIS import S3DIS, collation_fn, collation_fn_eval_all
        _ = S3DIS(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split='train', aug=args.aug,
                         memCacheInit=True, loop=args.loop)
        if args.evaluate:
            _ = S3DIS(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split='val', aug=False,
                             memCacheInit=True, eval_all=True)

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    global best_iou
    args = argss

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    model, label_model = get_model(args)

    if main_process():
        global logger, writer

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        logger = get_logger(args)
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)

    # ####################### Optimizer ####################### #

    optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': args.base_lr}], lr=args.base_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)


    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
    else:
        model = model.cuda()

    if label_model is not None:
        label_model = label_model.cuda()
        label_model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda(gpu)

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))
    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # schedular for scannet dataset
    if args.data_name == 'scannet':
        schedular = lr_scheduler.MultiStepLR(optimizer, milestones=[int(100 * 0.6), int(100 * 0.8)],
                                            gamma=0.1, last_epoch=args.start_epoch)  # args.epoch


    # ####################### Data Loader ####################### #
    if args.data_name == 's3dis':
        from dataset.S3DIS import S3DIS, collation_fn, collation_fn_eval_all
        train_data = S3DIS(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split='train', aug=args.aug,
                           memCacheInit=True, loop=args.loop, voxel_max=args.voxel_max, test_area=args.test_area)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if args.distributed else None
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                                   drop_last=True, collate_fn=collation_fn,
                                                   worker_init_fn=worker_init_fn)
        if args.evaluate:
            val_data = S3DIS(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split='val', aug=False,
                             memCacheInit=True, eval_all=True, test_area=args.test_area)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data) if args.distributed else None
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
                                                     shuffle=False, num_workers=args.workers, pin_memory=True,
                                                     drop_last=False, collate_fn=collation_fn_eval_all,
                                                     sampler=val_sampler)
    elif args.data_name == 'scannet':
        from dataset.scanNet3D import ScanNet3D, collation_fn, collation_fn_eval_all
        train_data = ScanNet3D(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split=args.train_split, aug=args.aug,
                               memCacheInit=True, loop=args.loop, voxel_max=args.voxel_max)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if args.distributed else None
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                                   drop_last=True, collate_fn=collation_fn,
                                                   worker_init_fn=worker_init_fn)
        if args.evaluate:
            val_data = ScanNet3D(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split='val', aug=False,
                                 memCacheInit=True, eval_all=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data) if args.distributed else None
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
                                                     shuffle=False, num_workers=args.workers, pin_memory=True,
                                                     drop_last=False, collate_fn=collation_fn_eval_all,
                                                     sampler=val_sampler)


    else:
        raise Exception('Dataset not supported yet'.format(args.data_name))

    # ####################### Train ####################### #
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            if args.evaluate:
                val_sampler.set_epoch(epoch)

        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, label_model, criterion, optimizer, epoch, args)

        if args.data_name == 'scannet':
            schedular.step()
            

        epoch_log = epoch + 1
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):

            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)

                # remember best iou and save checkpoint
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)
                print('learning rate:', optimizer.param_groups[0]['lr'])
                logger.info('==>Current best mIoU %.3f' % (best_iou))

        if (epoch_log % args.save_freq == 0) and main_process():
            model_path = os.path.join(args.save_path, 'model')
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            save_checkpoint(
                {
                    'epoch': epoch_log,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_iou': best_iou
                }, is_best, os.path.join(args.save_path, 'model')
            )
    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def get_model(cfg):
    if cfg.arch == 'mink':
        from models.unet_3d import MinkUNet18A as Model
        model = Model(in_channels=3, out_channels=cfg.classes, D=3)
        label_model = None
    elif cfg.arch == 'retro_fpn':
        from models.retrofpn import MinkUNet18A as Model
        from models.retrofpn import LabelNet as LabelModel
        model = Model(in_channels=3, 
                      out_channels=cfg.classes, D=3, 
                      plane_cls=cfg.plane_cls, 
                      plane_share=cfg.plane_share, 
                      strides=cfg.strides)
        label_model = LabelModel(plane_cls=cfg.classes + (0 if cfg.data_name == 's3dis' else 1), D=3)
    else:
        raise Exception('architecture not supported yet'.format(cfg.arch))
    return model, label_model


def label2Matrix(labels, num_classes):
    EYE = torch.eye(num_classes).cuda()
    return EYE[labels]


def train(train_loader, model, label_model, criterion, optimizer, epoch, args):
    torch.backends.cudnn.enabled = True
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        (coords, feat, label, batch_n) = batch_data
        # For some networks, making the network invariant to even, odd coords is important
        coords[:, 1:4] += (torch.rand(3) * 100).type_as(coords)

        feat = feat.cuda(non_blocking=True)
        coords = coords.cuda(non_blocking=True)
        sinput = SparseTensor(feat, coords)
        label = label.cuda(non_blocking=True)
        if args.arch == 'mink':
            output = model(sinput)
            loss = criterion(output, label)
        elif args.arch == 'retro_fpn':

            if args.data_name == 's3dis':
                label_mat = label2Matrix(label, args.classes)
            elif args.data_name == 'scannet':
                label_mat = label2Matrix(label, args.classes+1)  # label 255 has been turn to 20

            f1, f2, f3, f4, f5, out_1, idx2, idx3, idx4, idx5 = model(sinput, train=True)
            with torch.no_grad():
                label2, label3, label4, label5 = label_model(label_mat, out_1, idx2, idx3, idx4, idx5)

            output = f1
            loss1 = criterion(f1, label)
            loss2 = criterion(f2, label2)
            loss3 = criterion(f3, label3)
            loss4 = criterion(f4, label4)
            loss5 = criterion(f5, label5)

            if args.data_name == 's3dis':
                loss = loss1 + loss2 + loss3 + loss4 + loss5
            else:
                loss = (loss1*5 + loss2*4 + loss3*3 + loss4*2 + loss5) * 0.4
        else:
            raise Exception('architecture not supported yet'.format(args.arch))
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.detach().max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, label.detach(), args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), args.batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + i + 1
        current_lr = optimizer.param_groups[0]['lr']

        # Adjust lr for s3dis
        if args.data_name == 's3dis':
            current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)
            writer.add_scalar('learning_rate', current_lr, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch + 1, args.epochs, mIoU,
                                                                                           mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    torch.backends.cudnn.enabled = False  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            (coords, feat, label, inds_reverse, batch_n) = batch_data
            sinput = SparseTensor(feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
            label = label.cuda(non_blocking=True)
            output = model(sinput)
            # pdb.set_trace()
            output = output[inds_reverse, :]
            loss = criterion(output, label)

            output = output.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output, label.detach(), args.classes,
                                                                  args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            loss_meter.update(loss.item(), args.batch_size)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc




if __name__ == '__main__':
    main()
