import os
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import torch
import torch.nn as nn
import math
import argparse
import spectral
import torch
import torch.nn.parallel
from torch.utils.data.dataset import Dataset
from functools import partial
from itertools import repeat
from torch._six import container_abcs

import logging
from collections import OrderedDict
import scipy
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import time
from timm.models.layers import DropPath, trunc_normal_


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=True,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm2d(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, h * w], 1)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)

        return q, k, v

    def forward(self, x, h, w):
        if (
                self.conv_proj_q is not None
                or self.conv_proj_k is not None
                or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, h, w)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        flops = 0

        _, T, C = input.shape
        H = W = int(np.sqrt(T - 1)) if module.with_cls_token else int(np.sqrt(T))

        H_Q = H / module.stride_q
        W_Q = H / module.stride_q
        T_Q = H_Q * W_Q + 1 if module.with_cls_token else H_Q * W_Q

        H_KV = H / module.stride_kv
        W_KV = W / module.stride_kv
        T_KV = H_KV * W_KV + 1 if module.with_cls_token else H_KV * W_KV

        # C = module.dim
        # S = T
        # Scaled-dot-product macs
        # [B x T x C] x [B x C x T] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        flops += T_Q * T_KV * module.dim
        # [B x T x S] x [B x S x C] --> [B x T x C]
        flops += T_Q * module.dim * T_KV

        if (
                hasattr(module, 'conv_proj_q')
                and hasattr(module.conv_proj_q, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_q.conv.parameters()
                ]
            )
            flops += params * H_Q * W_Q

        if (
                hasattr(module, 'conv_proj_k')
                and hasattr(module.conv_proj_k, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_k.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        if (
                hasattr(module, 'conv_proj_v')
                and hasattr(module.conv_proj_v, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_v.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        params = sum([p.numel() for p in module.proj_q.parameters()])
        flops += params * T_Q
        params = sum([p.numel() for p in module.proj_k.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj_v.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj.parameters()])
        flops += params * T

        module.__flops__ += flops


class Block(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        self.with_cls_token = kwargs['with_cls_token']

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop,
            **kwargs
        )

        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x, h, w):
        res = x

        x = self.norm1(x)
        attn = self.attn(x, h, w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class ConvEmbed(nn.Module):
    """ Image to Conv Embedding

    """

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.rearrage = None

        self.patch_embed = ConvEmbed(
            # img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        with_cls_token = kwargs['with_cls_token']
        if with_cls_token:
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, embed_dim)
            )
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.size()

        x = rearrange(x, 'b c h w -> b (h w) c')

        cls_tokens = None
        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)

        if self.cls_token is not None:
            cls_tokens, x = torch.split(x, [1, H * W], 1)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x, cls_tokens


class ConvolutionalVisionTransformer(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()
        self.num_classes = num_classes
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.GELU(),
        )
        model = [
            {
                'patch_size': 3,
                'patch_stride': 1,
                'patch_padding': 0,
                'embed_dim': 64,
                'depth': 1,
                'num_heads': 1,
                'mlp_ratio': 4.0,
                'qkv_bias': True,
                'drop_rate': 0.0,
                'attn_drop_rate': 0.0,
                'drop_path_rate': 0.0,
                'with_cls_token': False,
                'method': 'dw_bn',
                'kernel_size': 3,
                'padding_q': 1,
                'padding_kv': 1,
                'stride_kv': 2,
                'stride_q': 1
            },
            {
                'patch_size': 3,
                'patch_stride': 1,
                'patch_padding': 0,
                'embed_dim': 108,
                'depth': 3,
                'num_heads': 3,
                'mlp_ratio': 4.0,
                'qkv_bias': True,
                'drop_rate': 0.0,
                'attn_drop_rate': 0.0,
                'drop_path_rate': 0.0,
                'with_cls_token': False,
                'method': 'dw_bn',
                'kernel_size': 3,
                'padding_q': 1,
                'padding_kv': 1,
                'stride_kv': 2,
                'stride_q': 1
            },
            {
                'patch_size': 3,
                'patch_stride': 1,
                'patch_padding': 0,
                'embed_dim': 204,
                'depth': 6,
                'num_heads': 6,
                'mlp_ratio': 4.0,
                'qkv_bias': True,
                'drop_rate': 0.0,
                'attn_drop_rate': 0.0,
                'drop_path_rate': 0.0,
                'with_cls_token': False,
                'method': 'dw_bn',
                'kernel_size': 3,
                'padding_q': 1,
                'padding_kv': 1,
                'stride_kv': 2,
                'stride_q': 1
            }
        ]
        self.num_stages = len(model)
        for i in range(self.num_stages):
            kwargs = {
                "patch_size": model[i]["patch_size"],
                "patch_stride": model[i]["patch_stride"],
                "patch_padding": model[i]["patch_padding"],
                "embed_dim": model[i]["embed_dim"],
                "depth": model[i]["depth"],
                "num_heads": model[i]["num_heads"],
                "mlp_ratio": model[i]["mlp_ratio"],
                "qkv_bias": model[i]["qkv_bias"],
                "drop_rate": model[i]["drop_rate"],
                "attn_drop_rate": model[i]["attn_drop_rate"],
                "drop_path_rate": model[i]["drop_path_rate"],
                "with_cls_token": model[i]["with_cls_token"],
                "method": model[i]["method"]
            }

            stage = VisionTransformer(
                in_chans=in_chans, init=init, act_layer=act_layer, norm_layer=norm_layer, **kwargs
            )
            setattr(self, f"stage{i}", stage)

            in_chans = model[i]["embed_dim"]

        dim_embed = model[-1]["embed_dim"]
        self.norm = norm_layer(dim_embed)
        self.cls_token = model[-1]['with_cls_token']

        # Classifier head
        self.head = nn.Linear(dim_embed, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.head.weight, std=0.02)

    def forward_features(self, x):
        # y = 32
        for i in range(self.num_stages):
            if x.shape[1] == 32:
                x, cls_tokens = getattr(self, f'stage{i}')(x)
            else:
                # mid = x
                x, cls_tokens = getattr(self, f'stage{i}')(x)
                # x = mid + x
            # padding = torch.autograd.Variable(torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).cuda())
            # print(padding.shape)
            # y = y * 2
            # mid = torch.cat((x, padding), 1)
            # print(mid.shape)
            # print(x.shape)
            # print(x.shape)

        if self.cls_token:
            x = self.norm(cls_tokens)
            x = torch.squeeze(x)
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)
            x = torch.mean(x, dim=1)

        return x

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.conv3d(x)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        x = self.forward_features(x)
        x = self.head(x)

        #         x = self.forward_features(x)
        #         x = self.head(x)

        return x


def random_unison(a, b, rstate=None):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p], b[p]


def split_data_fix(pixels, labels, n_samples, rand_state=None):
    pixels_number = np.unique(labels, return_counts=1)[1]
    train_set_size = [n_samples] * len(np.unique(labels))
    tr_size = int(sum(train_set_size))
    te_size = int(sum(pixels_number)) - int(sum(train_set_size))
    sizetr = np.array([tr_size] + list(pixels.shape)[1:])
    sizete = np.array([te_size] + list(pixels.shape)[1:])
    train_x = np.empty((sizetr));
    train_y = np.empty((tr_size));
    test_x = np.empty((sizete));
    test_y = np.empty((te_size))
    trcont = 0;
    tecont = 0;
    for cl in np.unique(labels):
        pixels_cl = pixels[labels == cl]
        labels_cl = labels[labels == cl]
        pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
        for cont, (a, b) in enumerate(zip(pixels_cl, labels_cl)):
            if cont < train_set_size[cl]:
                train_x[trcont, :, :, :] = a
                train_y[trcont] = b
                trcont += 1
            else:
                test_x[tecont, :, :, :] = a
                test_y[tecont] = b
                tecont += 1
    train_x, train_y = random_unison(train_x, train_y, rstate=rand_state)
    return train_x, test_x, train_y, test_y


def split_data(pixels, labels, percent, splitdset="custom", rand_state=69):
    splitdset = "sklearn"
    if splitdset == "sklearn":
        return train_test_split(pixels, labels, test_size=(1 - percent), stratify=labels, random_state=rand_state)
    elif splitdset == "custom":
        pixels_number = np.unique(labels, return_counts=1)[1]
        train_set_size = [int(np.ceil(a * percent)) for a in pixels_number]
        tr_size = int(sum(train_set_size))
        te_size = int(sum(pixels_number)) - int(sum(train_set_size))
        sizetr = np.array([tr_size] + list(pixels.shape)[1:])
        sizete = np.array([te_size] + list(pixels.shape)[1:])
        train_x = np.empty((sizetr));
        train_y = np.empty((tr_size));
        test_x = np.empty((sizete));
        test_y = np.empty((te_size))
        trcont = 0;
        tecont = 0;
        for cl in np.unique(labels):
            pixels_cl = pixels[labels == cl]
            labels_cl = labels[labels == cl]
            pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
            for cont, (a, b) in enumerate(zip(pixels_cl, labels_cl)):
                if cont < train_set_size[cl]:
                    train_x[trcont, :, :, :] = a
                    train_y[trcont] = b
                    trcont += 1
                else:
                    test_x[tecont, :, :, :] = a
                    test_y[tecont] = b
                    tecont += 1
        train_x, train_y = random_unison(train_x, train_y, rstate=rand_state)
        return train_x, test_x, train_y, test_y


def loadData(name, num_components=None):
    data_path = os.path.join(os.getcwd(), 'mnt')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SV':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']

    else:
        print("NO DATASET")
        exit()

    shapeor = data.shape
    data = data.reshape(-1, data.shape[-1])
    if num_components != None:
        data = PCA(n_components=num_components).fit_transform(data)
        shapeor = np.array(shapeor)
        shapeor[-1] = num_components
    # data = MinMaxScaler().fit_transform(data)
    data = StandardScaler().fit_transform(data)
    data = data.reshape(shapeor)
    num_class = len(np.unique(labels)) - 1
    return data, labels, num_class


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            # import matplotlib.pyplot as plt
            # plt.imshow(patch[:, :, 100])
            # plt.show()
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels.astype("int")


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(y_pred, y_test, name):
    classification = classification_report(y_test, y_pred, digits=6)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)

    return classification, confusion, list(np.round(np.array([oa, aa, kappa] + list(each_acc)) * 100, 2))


class HyperData(Dataset):
    def __init__(self, dataset):
        self.data = dataset[0].astype(np.float32)
        self.labels = []
        for n in dataset[1]: self.labels += [int(n)]

    def __getitem__(self, index):
        img = torch.from_numpy(np.asarray(self.data[index, :, :, :]))
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)

    def __labels__(self):
        return self.labels


def load_hyper(args):
    data, labels, numclass = loadData(args.dataset, num_components=args.components)
    pixels, labels = createImageCubes(data, labels, windowSize=args.spatialsize, removeZeroLabels=True)
    bands = pixels.shape[-1];
    numberofclass = len(np.unique(labels))
    if args.tr_percent < 1:  # split by percent
        x_train, x_test, y_train, y_test = split_data(pixels, labels, args.tr_percent)
    else:  # split by samples per class
        x_train, x_test, y_train, y_test = split_data_fix(pixels, labels, args.tr_percent)
    if args.use_val: x_val, x_test, y_val, y_test = split_data(x_test, y_test, args.val_percent)
    del pixels, labels
    train_hyper = HyperData((np.transpose(x_train, (0, 3, 1, 2)).astype("float32"), y_train))
    test_hyper = HyperData((np.transpose(x_test, (0, 3, 1, 2)).astype("float32"), y_test))
    if args.use_val:
        val_hyper = HyperData((np.transpose(x_val, (0, 3, 1, 2)).astype("float32"), y_val))
    else:
        val_hyper = None
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=args.tr_bsize, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_hyper, batch_size=args.te_bsize, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_hyper, batch_size=args.te_bsize, shuffle=False, **kwargs)
    return train_loader, test_loader, val_loader, numberofclass, bands


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    model.train()
    accs = np.ones((len(trainloader))) * -1000.0
    losses = np.ones((len(trainloader))) * -1000.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses[batch_idx] = loss.item()
        accs[batch_idx] = accuracy(outputs.data, targets.data)[0].item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (np.average(losses), np.average(accs))


def test(testloader, model, criterion, epoch, use_cuda):
    model.eval()
    accs = np.ones((len(testloader))) * -1000.0
    losses = np.ones((len(testloader))) * -1000.0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        outputs = model(inputs)
        losses[batch_idx] = criterion(outputs, targets).item()
        accs[batch_idx] = accuracy(outputs.data, targets.data, topk=(1,))[0].item()
    return (np.average(losses), np.average(accs))


def predict(testloader, model, criterion, use_cuda):
    model.eval()
    predicted = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda: inputs = inputs.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        [predicted.append(a) for a in model(inputs).data.cpu().numpy()]
    return np.array(predicted)


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class FocalLoss(torch.nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


def main():
    parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run') #PU-100   KSC-200
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--components', default='50', type=int, help='dimensionality reduction')
    parser.add_argument('--dataset', default='PU', type=str, help='dataset (options: IP, PU, SV, KSC)')
    parser.add_argument('--tr_percent', default=0.03, type=float, help='samples of train set')    #PU 3%    KSC 20%
    parser.add_argument('--tr_bsize', default=100, type=int, help='mini-batch train size (default: 100)')
    parser.add_argument('--te_bsize', default=100, type=int, help='mini-batch test size (default: 100)')
    parser.add_argument('--spatialsize', dest='spatialsize', default=15, type=int, help='spatial-spectral patch dimension')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--use_val', action='store_true', help='Use validation set')
    parser.add_argument('--val_percent', default=0.1, type=float, help='samples of val set')
    parser.set_defaults(bottleneck=True)
    #parser.add_argument('-f', type=str, default="读取额外的参数")
    best_err1 = 100

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    train_loader, test_loader, val_loader, num_classes, n_bands = load_hyper(args)

    # Use CUDA
    use_cuda = torch.cuda.is_available()
    if use_cuda: torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = ConvolutionalVisionTransformer(in_chans=args.components, num_classes=num_classes)
    model = ConvolutionalVisionTransformer(in_chans=(args.components - 2) * 8, num_classes=num_classes)
    if use_cuda: model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    # criterion = FocalLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr)
    best_acc = -1
    time_start = time.time()  # 记录开始时间
    for epoch in range(args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        if args.use_val:
            test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)
        else:
            test_loss, test_acc = test(test_loader, model, criterion, epoch, use_cuda)

        print("EPOCH", epoch, "TRAIN LOSS", train_loss, "TRAIN ACCURACY", train_acc, end=',')
        print("LOSS", test_loss, "ACCURACY", test_acc)
        # save model
        if test_acc > best_acc:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, "best_model.pth.tar")
            best_acc = test_acc
    time_end = time.time()
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print('运行时间为：', time_sum)
    checkpoint = torch.load("best_model.pth.tar")
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    test_loss, test_acc = test(test_loader, model, criterion, epoch, use_cuda)
    print("FINAL:      LOSS", test_loss, "ACCURACY", test_acc)
    classification, confusion, results = reports(np.argmax(predict(test_loader, model, criterion, use_cuda), axis=1),
                                                 np.array(test_loader.dataset.__labels__()), args.dataset)
    print(classification, args.dataset, results)


#     X, y,classo = loadData(args.dataset, num_components=args.components)
#     height = y.shape[0]
#     width = y.shape[1]
#     PATCH_SIZE = args.spatialsize
#     outputs = np.zeros((height, width))
#     X = padWithZeros(X, PATCH_SIZE // 2)
#     # calculate the predicted image
#     outputs = np.zeros((height, width))
#     for i in range(height):
#         for j in range(width):
#             if int(y[i, j]) == 0:
#                 continue
#             else:
#                 image_patch = X[i:i + PATCH_SIZE, j:j + PATCH_SIZE, :]
#                 #print(image_patch.shape)
#                 image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2])
#                 X_test_image = torch.FloatTensor(image_patch.transpose(0, 3, 1, 2)).cuda()
#                 prediction = model(X_test_image)
#                 prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
#                 outputs[i][j] = prediction + 1
#         if i % 20 == 0:
#             print('... ... row ', i, ' handling ... ...')

#     oringal_image = spectral.imshow(classes=y, figsize=(7, 7))
#     predict_image = spectral.imshow(classes=outputs.astype(int), figsize=(7, 7))
#     #spectral.save_rgb(args.dataset + "对比-原始.jpg", y.astype(int), colors=spectral.spy_colors)
#     spectral.save_rgb(args.dataset + "3D-VIT 预测.jpg", outputs.astype(int), colors=spectral.spy_colors)


if __name__ == '__main__':
    main()
