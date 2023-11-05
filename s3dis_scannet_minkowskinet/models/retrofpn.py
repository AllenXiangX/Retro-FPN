import os
import torch
import struct
import numpy as np
import MinkowskiEngine as ME
from time import time
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from models.resnet_mink import ResNetBase
from torch import nn
from models.retro_transformer import RetroTransformer
from MinkowskiEngine import SparseTensor



class LabelNet(nn.Module):
    def __init__(self, plane_cls=48, D=3):
        super(LabelNet, self).__init__()

        self.plane_cls = plane_cls

        # self.conv0p1s1 = ME.MinkowskiConvolution(
            # plane_cls, plane_cls, kernel_size=5, dimension=D)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            plane_cls, plane_cls, kernel_size=2, stride=2, dimension=D)

        self.conv2p2s2 = ME.MinkowskiConvolution(
            plane_cls, plane_cls, kernel_size=2, stride=2, dimension=D)

        self.conv3p4s2 = ME.MinkowskiConvolution(
            plane_cls, plane_cls, kernel_size=2, stride=2, dimension=D)

        self.conv4p8s2 = ME.MinkowskiConvolution(
            plane_cls, plane_cls, kernel_size=2, stride=2, dimension=D)

        for param in self.parameters():
            param.requires_grad = False

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                m.kernel[:, :, :] = 0
                for i in range(self.plane_cls):
                    m.kernel[:, i, i] = 1

    def forward(self, slabel, out_1, idx2, idx3, idx4, idx5, debug=False):
        slabel1 = SparseTensor(features=slabel, coordinate_map_key=out_1.coordinate_map_key, coordinate_manager=out_1.coordinate_manager)
        slabel2 = self.conv1p1s2(slabel1)
        slabel3 = self.conv2p2s2(slabel2)
        slabel4 = self.conv3p4s2(slabel3)
        slabel5 = self.conv4p8s2(slabel4)


        # slabel = torch.argmax(slabel.F, dim=-1)
        slabel2 = torch.argmax(slabel2.F[idx2], dim=-1)
        slabel3 = torch.argmax(slabel3.F[idx3], dim=-1)
        slabel4 = torch.argmax(slabel4.F[idx4], dim=-1)
        slabel5 = torch.argmax(slabel5.F[idx5], dim=-1)

        return slabel2, slabel3, slabel4, slabel5


class RetroFPN(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3, plane_cls=32, n_samples=12, plane_share=12, strides=(1, 1, 1, 1, 1)):
        self.plane_cls = plane_cls
        self.plane_share = plane_share
        self.strides = strides
        self.n_samples = n_samples
        self.n_groups = plane_cls // plane_share
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)


        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])
        
        self.relu = ME.MinkowskiReLU(inplace=True)

        self.rt1 = RetroTransformer(in_channels=self.PLANES[-1], out_channels=self.plane_cls, n_groups=self.n_groups, n_samples=self.n_samples)
        self.rt2 = RetroTransformer(in_channels=self.PLANES[-2], out_channels=self.plane_cls, n_groups=self.n_groups, n_samples=self.n_samples)
        self.rt3 = RetroTransformer(in_channels=self.PLANES[-3], out_channels=self.plane_cls, n_groups=self.n_groups, n_samples=self.n_samples)
        self.rt4 = RetroTransformer(in_channels=self.PLANES[-4], out_channels=self.plane_cls, n_groups=self.n_groups, n_samples=self.n_samples)
        self.rt5 = RetroTransformer(in_channels=self.PLANES[-5], out_channels=self.plane_cls, n_groups=self.n_groups, n_samples=self.n_samples)

        self.cls1 = nn.Sequential(nn.ReLU(), nn.Linear(self.plane_cls, out_channels, bias=True))
        self.cls2 = nn.Sequential(nn.ReLU(), nn.Linear(self.plane_cls, out_channels, bias=True))
        self.cls3 = nn.Sequential(nn.ReLU(), nn.Linear(self.plane_cls, out_channels, bias=True))
        self.cls4 = nn.Sequential(nn.ReLU(), nn.Linear(self.plane_cls, out_channels, bias=True))
        self.cls5 = nn.Sequential(nn.ReLU(), nn.Linear(self.plane_cls, out_channels, bias=True))



    def forward(self, x, train=False, inference=False, debug=False):
        out = self.conv0p1s1(x)
        # print(out)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out_5 = self.block4(out)  

        out = self.convtr4p16s2(out_5)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out_4 = self.block5(out)   

        out = self.convtr5p8s2(out_4)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out_3 = self.block6(out)

        out = self.convtr6p4s2(out_3)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out_2 = self.block7(out)

        out = self.convtr7p2s2(out_2)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        out_1 = self.block8(out) 

        x1 = out_1.F 
        x2 = out_2.F
        x3 = out_3.F
        x4 = out_4.F
        x5 = out_5.F

        p1, offsets_1 = self.get_offsets(out_1.C)
        p2, offsets_2, idx2 = self.get_offsets_idx(out_2.C, self.strides[1], offsets_1)
        p3, offsets_3, idx3 = self.get_offsets_idx(out_3.C, self.strides[2], offsets_2)
        p4, offsets_4, idx4 = self.get_offsets_idx(out_4.C, self.strides[3], offsets_3)
        p5, offsets_5, idx5 = self.get_offsets_idx(out_5.C, self.strides[4], offsets_4)

        x2 = x2[idx2]
        x3 = x3[idx3]
        x4 = x4[idx4]
        x5 = x5[idx5]

        fx5 = self.rt5([p5, x5, offsets_5])
        fx4 = self.rt4([p4, x4, offsets_4], [p5, fx5, offsets_5])
        fx3 = self.rt3([p3, x3, offsets_3], [p4, fx4, offsets_4])
        fx2 = self.rt2([p2, x2, offsets_2], [p3, fx3, offsets_3])
        fx1 = self.rt1([p1, x1, offsets_1], [p2, fx2, offsets_2])

        f5 = self.cls5(fx5)
        f4 = self.cls4(fx4)
        f3 = self.cls3(fx3)
        f2 = self.cls2(fx2)
        f1 = self.cls1(fx1)

        if train:
            return f1, f2, f3, f4, f5, out_1, idx2, idx3, idx4, idx5
        else:
            return f1


    def get_offsets(self, coords):
        p = coords[:, 1:].contiguous().float()
        batch_n = coords[:, 0].contiguous()
        min_, max_ = batch_n.min().item(), batch_n.max().item()

        sum_ = 0
        offset = []
        for n in range(min_, max_+1):
            sum_ += (batch_n == n).sum().item()
            offset.append(sum_)

        return p, torch.Tensor(offset).cuda().int()

    def get_offsets_idx(self, coords, stride, offsets_prev):
        # p = coords[:, 1:]  # .contiguous().float()
        batch_n = coords[:, 0]  # .contiguous()
        min_, max_ = batch_n.min().item(), batch_n.max().item()

        sum_vs = 0
        sum_vs_sampled = 0
        offset = []
        list_idx = []
        for i, b_i in enumerate(range(min_, max_ + 1)):
            num_prev = (offsets_prev[i] - (0 if i == 0 else offsets_prev[i-1])).item()  # number of points in previous layer
            num_strided = num_prev // stride
            num_curr = (batch_n == b_i).sum().item()  # number of voxel in current layer

            num_sample = min(num_curr, num_strided)  # numer of voxel to be sampled from current layer
            idx = torch.randperm(num_curr)[:num_sample].cuda() + sum_vs
            list_idx.append(idx)

            sum_vs += num_curr
            sum_vs_sampled += num_sample

            offset.append(sum_vs_sampled)

        idx_all = torch.cat(list_idx)

        return coords[idx_all, 1:].float(), torch.Tensor(offset).cuda().int(), idx_all


class MinkUNet14(RetroFPN):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class MinkUNet18(RetroFPN):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class MinkUNet34(RetroFPN):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet50(RetroFPN):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet101(RetroFPN):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class MinkUNet14A(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet14B(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet14C(MinkUNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class MinkUNet14D(MinkUNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet18A(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)
            # 32, 64, 128, 256, 256, 128, 96, 96

class MinkUNet18B(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet18D(MinkUNet18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet34A(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class MinkUNet34B(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class MinkUNet34C(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


def mink_unet(in_channels=3, out_channels=20, D=3, arch='MinkUNet18A'):
    if arch == 'MinkUNet18A':
        return MinkUNet18A(in_channels, out_channels, D)
    elif arch == 'MinkUNet18B':
        return MinkUNet18B(in_channels, out_channels, D)
    elif arch == 'MinkUNet18D':
        return MinkUNet18D(in_channels, out_channels, D)
    elif arch == 'MinkUNet34A':
        return MinkUNet34A(in_channels, out_channels, D)
    elif arch == 'MinkUNet34B':
        return MinkUNet34B(in_channels, out_channels, D)
    elif arch == 'MinkUNet34C':
        return MinkUNet34C(in_channels, out_channels, D)
    elif arch == 'MinkUNet14A':
        return MinkUNet14A(in_channels, out_channels, D)
    elif arch == 'MinkUNet14B':
        return MinkUNet14B(in_channels, out_channels, D)
    elif arch == 'MinkUNet14C':
        return MinkUNet14C(in_channels, out_channels, D)
    elif arch == 'MinkUNet14D':
        return MinkUNet14D(in_channels, out_channels, D)
    else:
        raise Exception('architecture not supported yet'.format(arch))
