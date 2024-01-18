from copy import deepcopy
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr

import einops
from timm.models.layers import DropPath
import pointops


from pcr.models.builder import MODELS
from pcr.models.utils import offset2batch, batch2offset

from .transformer import RetroTransformer

class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return self.norm(input.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


class GroupedVectorAttention(nn.Module):
    def __init__(self,
                 embed_channels,
                 groups,
                 attn_drop_rate=0.,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True
                 ):
        super(GroupedVectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )

        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups)
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat, coord, reference_index):
        query, key, value = self.linear_q(feat), self.linear_k(feat), self.linear_v(feat)
        key = pointops.grouping(reference_index, key, coord, with_xyz=True)
        value = pointops.grouping(reference_index, value, coord, with_xyz=False)
        pos, key = key[:, :, 0:3], key[:, :, 3:]
        relation_qk = key - query.unsqueeze(1)
        if self.pe_multiplier:
            pem = self.linear_p_multiplier(pos)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb
            value = value + peb

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(self.softmax(weight))

        mask = torch.sign(reference_index + 1)
        weight = torch.einsum("n s g, n s -> n s g", weight, mask)
        value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups)
        feat = torch.einsum("n s g i, n s g -> n g i", value, weight)
        feat = einops.rearrange(feat, "n g i -> n (g i)")
        return feat


class Block(nn.Module):
    def __init__(self,
                 embed_channels,
                 groups,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False
                 ):
        super(Block, self).__init__()
        self.attn = GroupedVectorAttention(
            embed_channels=embed_channels,
            groups=groups,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias
        )
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, points, reference_index):
        coord, feat, offset, _ = points
        identity = feat
        feat = self.act(self.norm1(self.fc1(feat)))
        feat = self.attn(feat, coord, reference_index) \
            if not self.enable_checkpoint else checkpoint(self.attn, feat, coord, reference_index)
        feat = self.act(self.norm2(feat))
        feat = self.norm3(self.fc3(feat))
        feat = identity + self.drop_path(feat)
        feat = self.act(feat)

        return [coord, feat, offset, _]


class BlockSequence(nn.Module):
    def __init__(self,
                 depth,
                 embed_channels,
                 groups,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False
                 ):
        super(BlockSequence, self).__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0. for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                embed_channels=embed_channels,
                groups=groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                enable_checkpoint=enable_checkpoint
            )
            self.blocks.append(block)

    def forward(self, points):

        coord, offset = points[0], points[2]
        # reference index query of neighbourhood attention
        # for windows attention, modify reference index query method
        reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)
        for block in self.blocks:
            points = block(points, reference_index)
        return points


class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 grid_size,
                 bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        coord, feat, offset, label = points
        batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))
        start = segment_csr(coord, torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                            reduce="min") if start is None else start
        cluster = voxel_grid(pos=coord - start[batch], size=self.grid_size, batch=batch, start=0)
        unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        label = None if label is None else segment_csr(label[sorted_cluster_indices], idx_ptr, reduce="sum")
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return [coord, feat, offset, label], cluster


class UnpoolWithSkip(nn.Module):
    """
    Map Unpooling with skip connection
    """

    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 bias=True,
                 skip=True,
                 backend="map"
                 ):
        super(UnpoolWithSkip, self).__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.skip = skip
        self.backend = backend
        assert self.backend in ["map", "interp"]

        self.proj = nn.Sequential(nn.Linear(in_channels, out_channels, bias=bias),
                                  PointBatchNorm(out_channels),
                                  nn.ReLU(inplace=True))
        self.proj_skip = nn.Sequential(nn.Linear(skip_channels, out_channels, bias=bias),
                                       PointBatchNorm(out_channels),
                                       nn.ReLU(inplace=True))

    def forward(self, points, skip_points, cluster=None):
        coord, feat, offset, _ = points
        skip_coord, skip_feat, skip_offset, _2 = skip_points
        if self.backend == "map" and cluster is not None:
            feat = self.proj(feat)[cluster]
        else:
            feat = pointops.interpolation(coord, skip_coord, self.proj(feat), offset, skip_offset)
        if self.skip:
            feat = feat + self.proj_skip(skip_feat)
        return [skip_coord, feat, skip_offset, _]


class Encoder(nn.Module):
    def __init__(self,
                 depth,
                 in_channels,
                 embed_channels,
                 groups,
                 grid_size=None,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=None,
                 drop_path_rate=None,
                 enable_checkpoint=False,
                 ):
        super(Encoder, self).__init__()

        self.down = GridPool(
            in_channels=in_channels,
            out_channels=embed_channels,
            grid_size=grid_size,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.,
            enable_checkpoint=enable_checkpoint
        )

    def forward(self, points):
        points, cluster = self.down(points)
        return self.blocks(points), cluster


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 skip_channels,
                 embed_channels,
                 groups,
                 depth,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=None,
                 drop_path_rate=None,
                 enable_checkpoint=False,
                 unpool_backend="map"
                 ):
        super(Decoder, self).__init__()

        self.up = UnpoolWithSkip(
            in_channels=in_channels,
            out_channels=embed_channels,
            skip_channels=skip_channels,
            backend=unpool_backend
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.,
            enable_checkpoint=enable_checkpoint
        )

    def forward(self, points, skip_points, cluster):
        points = self.up(points, skip_points, cluster)
        return self.blocks(points)


class GVAPatchEmbed(nn.Module):
    def __init__(self,
                 depth,
                 in_channels,
                 embed_channels,
                 groups,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False
                 ):
        super(GVAPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.proj = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint
        )

    def forward(self, points):
        coord, feat, offset, _ = points
        feat = self.proj(feat)
        return self.blocks([coord, feat, offset, _])


@MODELS.register_module("retrofpn")
class RetroFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 patch_embed_depth=1,
                 patch_embed_channels=48,
                 patch_embed_groups=6,
                 patch_embed_neighbours=8,
                 enc_depths=(2, 2, 6, 2),
                 enc_channels=(96, 192, 384, 512),
                 enc_groups=(12, 24, 48, 64),
                 enc_neighbours=(16, 16, 16, 16),
                 dec_depths=(1, 1, 1, 1),
                 dec_channels=(48, 96, 192, 384),
                 dec_groups=(6, 12, 24, 48),
                 dec_neighbours=(16, 16, 16, 16),
                 grid_sizes=(0.06, 0.12, 0.24, 0.48),
                 attn_qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0,
                 enable_checkpoint=False,
                 unpool_backend="map"
                 ):
        super(RetroFPN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_depths)
        assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(dec_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(dec_neighbours)
        assert self.num_stages == len(grid_sizes)
        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint
        )

        self.plane_cls = 48
        self.n_groups = 4
        self.n_sample = 12
        self.strides = (1, 4, 1, 1, 1)
        self.n_samples = (12, 12, 16, 16, 12)
        print('plane_cls, ', self.plane_cls)
        print('n_groups, ', self.n_groups)
        print('n_sample, ', self.n_sample)
        print('strides, ', self.strides)
        print('n_samples, ', self.n_samples)
        self.rt1 = RetroTransformer(48, out_channels=self.plane_cls, n_groups=self.n_groups, n_samples=self.n_samples[0])
        self.rt2 = RetroTransformer(96, out_channels=self.plane_cls, n_groups=self.n_groups, n_samples=self.n_samples[1])
        self.rt3 = RetroTransformer(192, out_channels=self.plane_cls, n_groups=self.n_groups, n_samples=self.n_samples[2])
        self.rt4 = RetroTransformer(384, out_channels=self.plane_cls, n_groups=self.n_groups, n_samples=self.n_samples[3])
        self.rt5 = RetroTransformer(512, out_channels=self.plane_cls, n_groups=self.n_groups, n_samples=self.n_samples[4])

        self.cls1 = nn.Sequential(nn.ReLU(), nn.Linear(self.plane_cls, num_classes, bias=True))
        self.cls2 = nn.Sequential(nn.ReLU(), nn.Linear(self.plane_cls, num_classes, bias=True))
        self.cls3 = nn.Sequential(nn.ReLU(), nn.Linear(self.plane_cls, num_classes, bias=True))
        self.cls4 = nn.Sequential(nn.ReLU(), nn.Linear(self.plane_cls, num_classes, bias=True))
        self.cls5 = nn.Sequential(nn.ReLU(), nn.Linear(self.plane_cls, num_classes, bias=True))


        enc_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))]
        dec_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))]
        enc_channels = [patch_embed_channels] + list(enc_channels)
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[sum(enc_depths[:i]):sum(enc_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint
            )
            dec = Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                skip_channels=enc_channels[i],
                embed_channels=dec_channels[i],
                groups=dec_groups[i],
                neighbours=dec_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dec_dp_rates[sum(dec_depths[:i]):sum(dec_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint,
                unpool_backend=unpool_backend
            )
            self.enc_stages.append(enc)
            self.dec_stages.append(dec)
        # self.seg_head = nn.Sequential(
        #     nn.Linear(dec_channels[0], dec_channels[0]),
        #     PointBatchNorm(dec_channels[0]),
        #    nn.ReLU(inplace=True),
        #     nn.Linear(dec_channels[0], num_classes)
        # ) if num_classes > 0 else nn.Identity()

    def forward(self, data_dict, train=False, debug=False):
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"].int()
        if 'label_mat' in data_dict:
            label = data_dict["label_mat"]
        else:
            label = None
        label_2to5 = []

        # a batch of point cloud is a list of coord, feat and offset
        points = [coord, feat, offset, label]
        points = self.patch_embed(points)
        skips = [[points]]
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            skips[-1].append(cluster)  # record grid cluster of pooling
            label_2to5.append(None if label is None else torch.argmax(points[-1], dim=-1))  # argmax for point-level supervision [:, :-1], softmax for region level
            skips.append([points])  # record points info of current stage

        points = skips.pop(-1)[0]  # unpooling points info in the last enc stage
        # print(points[1].shape)
        list_pxo = [points[:-1]]
        for i in reversed(range(self.num_stages)):
            skip_points, cluster = skips.pop(-1)
            points = self.dec_stages[i](points, skip_points, cluster)
            # print(points[1].shape)
            list_pxo.append(points[:-1])
        # seg_logits = self.seg_head(feat)

        [p5, x5, o5] , idx_5 = self.random_sampling(list_pxo[0], self.strides[4])
        [p4, x4, o4] , idx_4 = self.random_sampling(list_pxo[1], self.strides[3])
        [p3, x3, o3] , idx_3 = self.random_sampling(list_pxo[2], self.strides[2])
        [p2, x2, o2] , idx_2 = self.random_sampling(list_pxo[3], self.strides[1])
        [p1, x1, o1] = list_pxo[4]  #

        fx5 = self.rt5([p5, x5, o5])
        fx4 = self.rt4([p4, x4, o4], [p5, fx5, o5])
        fx3 = self.rt3([p3, x3, o3], [p4, fx4, o4])
        fx2 = self.rt2([p2, x2, o2], [p3, fx3, o3])
        fx1 = self.rt1([p1, x1, o1], [p2, fx2, o2])

        f5 = self.cls5(fx5)
        f4 = self.cls4(fx4)
        f3 = self.cls3(fx3)
        f2 = self.cls2(fx2)
        f1 = self.cls1(fx1)

        if debug:
            label2 = label_2to5[0] # [idx_2]
            label3 = label_2to5[1] # [idx_3]
            label4 = label_2to5[2] # [idx_4]
            label5 = label_2to5[3] # [idx_5]
            # return f1, f2, f3, f4, f5, label2, label3, label4, label5

            return f1, f2, f3, f4, f5, \
                   label2, label3, label4, label5, o1, o2, o3, o4, o5, \
                   p1, p2, p3, p4, p5


        if train:
            # inds2 = sort_inds(out_2.C)
            # inds3 = sort_inds(out_3.C)
            # inds4 = sort_inds(out_4.C)
            # inds5 = sort_inds(out_5.C)

            # f2 = f2[inds2]
            # f3 = f3[inds3]
            # f4 = f4[inds4]
            # f5 = f5[inds5]
            # return f1, f2, f3, f4, f5
            label2 = label_2to5[0][idx_2]
            label3 = label_2to5[1][idx_3]
            label4 = label_2to5[2][idx_4]
            label5 = label_2to5[3][idx_5]
            return f1, f2, f3, f4, f5, label2, label3, label4, label5
        else:
            return f1

    def random_sampling(self, pxo, rate):
        if rate == 1:
            return pxo, torch.arange(pxo[0].shape[0]).cuda()
        p, x, o = pxo
        LEN = len(o)
        list_idx = []
        prev_idx = 0
        o2 = []
        for i in range(LEN):
            low = 0 if i == 0 else o[i - 1]
            high = o[i]
            n_sub = (high-low) // rate
            batch = torch.arange(low, high)[torch.randperm(high - low)].cuda()
            idx = batch[:n_sub]
            list_idx.append(idx)
            prev_idx = prev_idx + n_sub
            o2.append(prev_idx)

        idx_all = torch.cat(list_idx).cuda().long()

        p2 = p[idx_all]
        x2 = x[idx_all]
        o2 = torch.Tensor(o2).cuda().int()

        return [p2, x2, o2], idx_all
