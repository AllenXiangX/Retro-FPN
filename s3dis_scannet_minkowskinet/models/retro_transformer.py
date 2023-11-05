import math
import torch
from torch import nn

from lib.pointops.functions import pointops

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


class RetroTransformer(nn.Module):

    def __init__(self, in_channels, out_channels=32, n_groups=4, n_samples=16):
        super(RetroTransformer, self).__init__()

        self.mlp_bottleneck = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels, bias=False)
        )

        self.gav = CrossAttention(out_channels, n_groups=n_groups, n_samples=n_samples)

        self.mlp_out = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        self.linear_info_curr = nn.Linear(in_channels, out_channels)

        self.mlp_gate = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.Linear(out_channels, out_channels, bias=False),
            nn.Sigmoid()
        )


    def forward(self, pxo, pxo2=None):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        info_curr = self.linear_info_curr(x)

        x = self.mlp_bottleneck(x)

        x = self.gav([p, x, o], [p, x, o] if pxo2 is None else pxo2)

        x = self.mlp_out(x)

        z = self.mlp_gate(x + info_curr)

        x = (1 - z) * x + z * info_curr

        return [p, x, o]
