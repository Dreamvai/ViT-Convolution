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

# 用于测试样本的比例
test_ratio = 0.90
# 每个像素周围提取 patch 的尺寸
patch_size = 15
name = "IP"
if name == 'PU':
    class_num = 9
if name == 'SA':
    class_num = 16
if name == 'IP':
    class_num = 16
if name == 'KSC':
    class_num = 13
# 使用 PCA 降维，得到主成分的数量
if name == 'PU':
    pca_components = 50
if name == 'IP':
    pca_components = 100
if name == 'KSC':
    pca_components = 50

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
        else:
            q, k, v = self.forward_conv(x, h, w)
        #print('这是q的形状', q.shape)
        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)
        #print('这是分头之后q的形状', q.shape)
        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)
       # print('这是多头之后的', attn.shape)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        #print('这是多头之后的', x.shape)
        x = rearrange(x, 'b h t d -> b t (h d)')
        #print('这是多头之后的', x.shape)

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
                #'method': 'linear',
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
                #'method': 'linear',
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
                #'method': 'linear',
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
            #print('第一个', x.shape)
            #print('到底是多少', cls_tokens)
                # x = mid + x
            # padding = torch.autograd.Variable(torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).cuda())
            # print(padding.shape)
            # y = y * 2
            # mid = torch.cat((x, padding), 1)
            # print(mid.shape)
            # print(x.shape)
            # print(x.shape)

        if self.cls_token:
            #print('使用的cls.token')
            x = self.norm(cls_tokens)
            #print(x.shape)
            x = torch.squeeze(x)
            #print(x.shape)
        else:
            # print('不使用的cls.token')
            # print(x.shape)
            x = rearrange(x, 'b c h w -> b (h w) c')
            #print(x.shape)
            x = self.norm(x)
            #print(x.shape)
            x = torch.mean(x, dim=1)
            #print(x.shape)
        return x

    def forward(self, x):
        # x = torch.unsqueeze(x, dim=1)
        x = self.conv3d(x)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        x = self.forward_features(x)
        x = self.head(x)

        #         x = self.forward_features(x)
        #         x = self.head(x)

        return x


################################数据处理################################
# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


# # 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
# def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):#X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
#     # 给 X 做 padding
#     margin = int((windowSize - 1) / 2)
#     zeroPaddedX = padWithZeros(X, margin=margin)
#     # split patches
#     patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))##(145*145，25，25，30)
#     patchesLabels = np.zeros((X.shape[0] * X.shape[1]))#(145*145)
#     patchIndex = 0
#     for r in range(margin, zeroPaddedX.shape[0] - margin):
#         for c in range(margin, zeroPaddedX.shape[1] - margin):
#             patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
#             patchesData[patchIndex, :, :, :] = patch
#             patchesLabels[patchIndex] = y[r-margin, c-margin]
#             patchIndex = patchIndex + 1
#     if removeZeroLabels:
#         patchesData = patchesData[patchesLabels > 0, :, :, :]
#         patchesLabels = patchesLabels[patchesLabels > 0]
#         patchesLabels -= 1
#     return patchesData, patchesLabels
# 在每个像素周围提取 patch
def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # 获得 y 中的标记样本数
    count = 0
    for r in range(0, y.shape[0]):
        for c in range(0, y.shape[1]):
            if y[r, c] != 0:
                count = count + 1

    # split patches
    patchesData = np.zeros([count, windowSize, windowSize, X.shape[2]])
    patchesLabels = np.zeros(count)

    count = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            if y[r - margin, c - margin] != 0:
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[count, :, :, :] = patch
                patchesLabels[count] = y[r - margin, c - margin]
                count = count + 1

    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    return X_train, X_test, y_train, y_test
# 类别
# class_num = 16
# X = sio.loadmat('C:\AI\data\Indian_pines_corrected.mat')['indian_pines_corrected']#X是数据集
# y = sio.loadmat('C:\AI\data\Indian_pines_gt.mat')['indian_pines_gt']#y是标签
def loadData(name):
    data_path = "C:\AI\data"
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
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
    return data, labels
X, y = loadData(name)
print('Hyperspectral data shape: ', X.shape)  ##(145,145,200)
print('Label shape: ', y.shape)  # (145,145)

print('\n... ... PCA tranformation ... ...')
X_pca = applyPCA(X, numComponents=pca_components)
print('Data shape after PCA: ', X_pca.shape)  # (145,145,30)

print('\n... ... create data cubes ... ...')
X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)  # (10249,25,25,30)    (10249)
print('Data cube X shape: ', X_pca.shape)  # (10249,25,25,30)
print('Data cube y shape: ', y.shape)  # (10249)

print('\n... ... create train & test data ... ...')
Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio)
print('Xtrain shape: ', Xtrain.shape)  # (1024, 25, 25, 30)
print('Xtest  shape: ', Xtest.shape)  # (9225, 25, 25, 30)

# 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
print('before transpose: Xtrain shape: ', Xtrain.shape)  # (1024, 25, 25, 30, 1)
print('before transpose: Xtest  shape: ', Xtest.shape)  # (9225, 25, 25, 30, 1)

# 为了适应 pytorch 结构，数据要做 transpose
Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
Xtest = Xtest.transpose(0, 4, 3, 1, 2)
print('after transpose: Xtrain shape: ', Xtrain.shape)  # (1024, 1, 30, 25, 25)
print('after transpose: Xtest  shape: ', Xtest.shape)  # (9225, 1, 30, 25, 25)

""" Training dataset"""
class TrainDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len
""" Testing dataset"""
class TestDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len
# 创建 trainloader 和 testloader
trainset = TrainDS()
testset = TestDS()
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=128, shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
one = torch.ones(128, dtype=torch.long).to(device)
two = torch.ones(18, dtype=torch.long).to(device)
# 网络放到GPU上
# net = CNN_3D_old().to(device)
# net = CTMixer(channels=100, num_classes=16, image_size=15, datasetname='IndianPines', num_layers=1, num_heads=4).to(device)
net = ConvolutionalVisionTransformer(in_chans=(pca_components - 2) * 8, num_classes=class_num).to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)

# 开始训练
time_start = time.time()  # 记录开始时间
net.train()
total_loss = 0
for epoch in range(100):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # print(labels.shape)
        try:
            labels = labels - one
        except:
            labels = labels - two

        # print(labels)
        # 优化器梯度归零
        optimizer.zero_grad()
        # 正向传播 +　反向传播 + 优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1, total_loss / (epoch + 1), loss.item()))

print('Finished Training')
time_end = time.time()
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum)
path = 'model' + '.pth'
torch.save(net, path)

if name == 'PU':
    t = int(42776 * test_ratio) + 1  # 10249 42776 54129
if name == 'SA':
    t = int(54129 * test_ratio) + 1  # 10249 42776 54129
if name == 'IP':
    t = int(10249 * test_ratio) + 1  # 10249 42776 54129
if name == 'KSC':
    t = int(5211 * test_ratio) + 1  # 10249 42776 54129
# print(t)
a = np.ones(t)  ##9225是Xtest.shape[0]
ytest = ytest - a

count = 0
# 模型测试
net.eval()
for inputs, _ in test_loader:
    inputs = inputs.to(device)
    outputs = net(inputs)
    outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
    if count == 0:
        y_pred_test = outputs
        count = 1
    else:
        y_pred_test = np.concatenate((y_pred_test, outputs))

# 生成分类报告
classification = classification_report(ytest, y_pred_test, digits=6)
print(classification)

from operator import truediv


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(test_loader, y_test, name):
    count = 0
    # 模型测试
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred = outputs
            count = 1
        else:
            y_pred = np.concatenate((y_pred, outputs))

    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
    elif name == 'KSC':
        target_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '10', '12', '13']

    classification = classification_report(y_test, y_pred, target_names=target_names)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)

    return classification, confusion, oa * 100, each_acc * 100, aa * 100, kappa * 100


# 将结果写在文件里
classification, confusion, oa, each_acc, aa, kappa = reports(test_loader, ytest, name)
print('oa aa kappa:', oa, aa, kappa)
classification = str(classification)
confusion = str(confusion)
# file_name = "classification_report.txt"

# with open(file_name, 'w') as x_file:
#     x_file.write('\n')
#     x_file.write('{} Kappa accuracy (%)'.format(kappa))
#     x_file.write('\n')
#     x_file.write('{} Overall accuracy (%)'.format(oa))
#     x_file.write('\n')
#     x_file.write('{} Average accuracy (%)'.format(aa))
#     x_file.write('\n')
#     x_file.write('\n')
#     x_file.write('{}'.format(classification))
#     x_file.write('\n')
#     x_file.write('{}'.format(confusion))
# 显示结果
# load the original image


# X, y = loadData(name)
# height = y.shape[0]
# width = y.shape[1]

# X = applyPCA(X, numComponents=pca_components)
# X = padWithZeros(X, patch_size // 2)

# # 逐像素预测类别
# outputs = np.zeros((height, width))
# for i in range(height):
#     for j in range(width):
#         if int(y[i, j]) == 0:
#             continue
#         else:
#             image_patch = X[i:i + patch_size, j:j + patch_size, :]
#             image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2], 1)
#             X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
#             prediction = net(X_test_image)
#             prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
#             outputs[i][j] = prediction + 1
#     if i % 20 == 0:
#         print('... ... row ', i, ' handling ... ...')

# oringal_image = spectral.imshow(classes=y, figsize=(7, 7))
# predict_image = spectral.imshow(classes=outputs.astype(int), figsize=(7, 7))
# spectral.save_rgb(name + " HybridSN-原始.jpg", y.astype(int), colors=spectral.spy_colors)
# spectral.save_rgb(name + " HybridSN-预测.jpg", outputs.astype(int), colors=spectral.spy_colors)