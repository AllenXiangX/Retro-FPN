import sys
import math
import torch
import torch.nn as nn
from lib.pointops.functions import pointops
from model.retrofpn.retro_transformer import RetroTransformer


class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(
            nn.Conv1d(3, 3, 1),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Conv1d(3, out_planes, 1)
        )

        self.linear_w = nn.Sequential(
            nn.BatchNorm1d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_planes, mid_planes // share_planes, 1),
            nn.BatchNorm1d(mid_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_planes // share_planes, out_planes // share_planes, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k, idx = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True, return_idx=True)  # (n, nsample, 3+c)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, idx, o, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]

        p_r = p_r.transpose(1, 2).contiguous()
        p_r = self.linear_p(p_r)
        p_r = p_r.transpose(1, 2).contiguous()
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes,
                                              self.mid_planes).sum(2)  # (n, nsample, c)

        w = w.transpose(1, 2).contiguous()
        w = self.linear_w(w)
        w = w.transpose(1, 2).contiguous()
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape
        s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
            return [p, x, o], idx
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
            return [p, x, o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2 * in_planes, in_planes), nn.BatchNorm1d(in_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes),
                                         nn.ReLU(inplace=True))

    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        # x = self.relu(x)
        return [p, x, o]


class ResBlock(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128, if_bn=False):
        super(ResBlock, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.linear_1 = nn.Linear(in_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)
        self.linear_shortcut = nn.Linear(in_dim, out_dim)
        self.if_bn = if_bn
        if self.if_bn:
            self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.linear_shortcut(x)
        if self.if_bn:
            out = self.linear_2(torch.relu(self.bn(self.linear_1(x)))) + shortcut
        else:
            out = self.linear_2(torch.relu(self.linear_1(x))) + shortcut
        return out

class CrossAttention(nn.Module):
    def __init__(self, n_channels, n_groups, n_samples=16):
        super(CrossAttention, self).__init__()
        self.n_channels = n_channels
        self.n_groups = n_groups
        self.n_samples = n_samples
        self.n_share_channels = self.n_channels // self.n_groups

        self.linear_q = nn.Linear(n_channels, n_channels)
        self.linear_k = nn.Linear(n_channels, n_channels)
        self.linear_v = nn.Linear(n_channels, n_channels)


        self.linear_p = nn.Sequential(
            nn.Conv1d(3, 3, 1),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Conv1d(3, self.n_groups*2, 1)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo, pxo2):
        p, x, o = pxo  # (n, 3), (n, c), (b)  current layer
        p2, x2, o2 = pxo2  # (m, 3), (m, c), (b)  previous layer
        n = x.shape[0]

        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x2), self.linear_v(x2)  # (n, c)  (m, c)  # , self.linear_v(x)

        x_k, idx = pointops.queryandgroup2(self.n_samples, p2, p, x_k, None, o2, o)
        x_v = x_v[idx.view(-1).long(), :].view(n, self.n_samples, -1)  # n, n_samples, c
        p_r = p2[idx.view(-1).long(), :].view(n, self.n_samples, -1) - p.unsqueeze(1)

        p_r = p_r.transpose(1, 2)  # .contiguous()
        p_rq, p_rk = torch.chunk(self.linear_p(p_r).transpose(1, 2).unsqueeze(-1), 2, 2)  # n, n_samples, n_groups, 1

        x_k = x_k.view(n, self.n_samples, self.n_groups, self.n_share_channels) + p_rk
        x_q = x_q.unsqueeze(1).repeat((1, self.n_samples, 1)).view(n, self.n_samples, self.n_groups, self.n_share_channels) + p_rq

        w = self.softmax(torch.sum(x_k * x_q, dim=-1, keepdim=True) / math.sqrt(self.n_share_channels))  # n, n_samples, n_groups

        x = (x_v.view(n, self.n_samples, self.n_groups, self.n_share_channels) * w).sum(1).view(n, -1)
        return x


class RetroFPNSeg(nn.Module):
    def __init__(self, block, blocks, c=6, k=13, plane_cls=32, n_samples_retro=16):
        super().__init__()
        self.c = c
        self.n_groups = 1
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0],
                                   nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1],
                                   nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2],
                                   nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3],
                                   nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4],
                                   nsample=nsample[4])  # N/256

        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.rt5 = RetroTransformer(planes[4], out_channels=plane_cls, n_groups=self.n_groups)
        self.cls5 = nn.Sequential(nn.BatchNorm1d(plane_cls), nn.ReLU(), nn.Linear(plane_cls, k))

        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.rt4 = RetroTransformer(planes[3], out_channels=plane_cls, n_groups=self.n_groups)
        self.cls4 = nn.Sequential(nn.BatchNorm1d(plane_cls), nn.ReLU(), nn.Linear(plane_cls, k))


        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.rt3 = RetroTransformer(planes[2], out_channels=plane_cls, n_groups=self.n_groups)
        self.cls3 = nn.Sequential(nn.BatchNorm1d(plane_cls), nn.ReLU(), nn.Linear(plane_cls, k))


        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.rt2 = RetroTransformer(planes[1], out_channels=plane_cls, n_groups=self.n_groups)
        self.cls2 = nn.Sequential(nn.BatchNorm1d(plane_cls), nn.ReLU(), nn.Linear(plane_cls, k))


        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        self.rt1 = RetroTransformer(planes[0], out_channels=plane_cls, n_groups=self.n_groups)
        self.cls1 = nn.Sequential(nn.BatchNorm1d(plane_cls), nn.ReLU(), nn.Linear(plane_cls, k))


    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, pxo, train=False, target=None, criterion=None, inference=False):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])

        [p2, x2, o2], idx2 = self.enc2[0]([p1, x1, o1])
        p2, x2, o2 = self.enc2[1:]([p2, x2, o2])

        [p3, x3, o3], idx3 = self.enc3[0]([p2, x2, o2])
        p3, x3, o3 = self.enc3[1:]([p3, x3, o3])

        [p4, x4, o4], idx4 = self.enc4[0]([p3, x3, o3])
        p4, x4, o4 = self.enc4[1:]([p4, x4, o4])

        [p5, x5, o5], idx5 = self.enc5[0]([p4, x4, o4])
        p5, x5, o5 = self.enc5[1:]([p5, x5, o5])

        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]

        fx5 = self.rt5([p5, x5, o5])[1]

        fx4 = self.rt4([p4, x4, o4], [p5, fx5, o5])[1]

        fx3 = self.rt3([p3, x3, o3], [p4, fx4, o4])[1]

        fx2 = self.rt2([p2, x2, o2], [p3, fx3, o3])[1]

        fx1 = self.rt1([p1, x1, o1], [p2, fx2, o2])[1]

        x5 = self.cls5(fx5)
        x4 = self.cls4(fx4)
        x3 = self.cls3(fx3)
        x2 = self.cls2(fx2)
        x1 = self.cls1(fx1)

        if inference:
            return x1, x2, x3, x4, x5, idx2, idx3, idx4, idx5, p1, p2, p3, p4, p5
        elif train:
            return x1, x2, x3, x4, x5, idx2, idx3, idx4, idx5
        else:
            return x1



def retrofpn_seg_repro(**kwargs):
    model = RetroFPNSeg(PointTransformerBlock, [2, 3, 4, 6, 3], **kwargs)
    return model


class Model(nn.Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__()
        self.model = RetroFPNSeg(PointTransformerBlock, [1, 1, 1, 1, 1], **kwargs)
        self.criterion = nn.CrossEntropyLoss()

    def pred(self, pcd_feat, cat_one_hot, offsets):
        p, x = pcd_feat[:, :3].contiguous(), pcd_feat
        pred = self.model([p, x, offsets], cat_one_hot)
        return pred

    def get_loss(self, pcd_feat, cat_one_hot, offsets, target):
        # target.unsqueeze(1)
        p, x = pcd_feat[:, :3].contiguous(), pcd_feat
        outputs, idxs = self.model([p, x, offsets], cat_one_hot, train=True)
        x1, x2, x3, x4, x5 = outputs
        idx2, idx3, idx4, idx5 = idxs
        target1 = target
        target2 = target1[idx2.long()]
        target3 = target2[idx3.long()]
        target4 = target3[idx4.long()]
        target5 = target4[idx5.long()]

        loss1 = self.criterion(x1, target1)
        loss2 = self.criterion(x2, target2)
        loss3 = self.criterion(x3, target3)
        loss4 = self.criterion(x4, target4)
        loss5 = self.criterion(x5, target5)

        return loss1 + loss2 + loss3 + loss4 + loss5
