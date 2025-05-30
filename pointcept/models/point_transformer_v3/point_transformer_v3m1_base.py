"""
Point Transformer - V3 Mode1

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential

# Token Merging Algorithms
from pointcept.models.point_transformer_v3.token_merging_algos import *

import math
from typing import Callable, Tuple

import torch

VALID_TOME_MODES = ["patch", "tome", "pool", "progressive", "pitome", "important_patch", "random_patch", "weighted_patch"]

class ValueReplace(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, x):
        return self.norm(x)

class PoolReplace(torch.nn.Module):
    def __init__(self, method, kernel_size, channels, fuse_original=False, layer_norm=False):
        super().__init__()
        pool_params = {
            "kernel_size": kernel_size,
            "stride": 1,
            "padding": kernel_size // 2,
        }
        self.fuse_original = fuse_original
        if method == "AvgPool1d":
            pool_params["count_include_pad"] = False
        self.pool = getattr(nn, method)(**pool_params)
        self.fuse_original = fuse_original
        if layer_norm:
            self.norm = nn.LayerNorm(channels)
        else:
            self.norm = None
    
    def forward(self, x):
        if self.norm:
            return self.norm(self.pool(x))
        if self.fuse_original:
            return (self.pool(x) + x) / 2
        return self.pool(x)


class ShufflePoolReplace(torch.nn.Module):
    def __init__(self, method, kernel_size, shuffle_steps, channels, fuse_original=False, layer_norm=False):
        super().__init__()
        self.shuffle_steps = shuffle_steps
        pool_params = {
            "kernel_size": kernel_size,
            "stride": 1,
            "padding": kernel_size // 2,
        }
        if method == "AvgPool1d":
            pool_params["count_include_pad"] = False
        self.pool = getattr(nn, method)(**pool_params)
        self.fuse_original = fuse_original
        if layer_norm:
            self.norm = nn.LayerNorm(channels)
        else: 
            self.norm = None
    
    def shuffle(self, x):
        # Shuffle [B, C, L] --> [B, C, L//shuffle_steps, shuffle_steps] --> [B, C, shuffle_steps, L//shuffle_steps] --> [B, C, L]
        B = x.shape[0]
        C = x.shape[1]
        shuffle_steps = self.shuffle_steps
        quotient = x.shape[2] // shuffle_steps
        remainder = x.shape[2] % shuffle_steps
        
        if quotient == 0:
            return x
        if x.shape[2] % shuffle_steps != 0:
            x1 = x[..., :-remainder] # [B, C, L1]
            x2 = x[..., -remainder:] # [B, C, L2]
            x1 = x1.reshape(B, C, x1.shape[2] // shuffle_steps, shuffle_steps) #[B, C, L1//shuffle_steps, shuffle_steps])
            x1 = x1.transpose(-1, -2) # [B, C, shuffle_steps, L1//shuffle_steps])
            x1a = x1[:, :, :remainder, :] # [B, C, remainder, L1//shuffle_steps]
            x1b = x1[:, :, remainder:, :] # [B, C, shuffle_steps - remainder, L1//shuffle_steps]
            x1ax2 = torch.cat([x1a, x2.unsqueeze(3)], dim=-1) # [B, C, remainder, L1//shuffle_steps + 1]
            x1ax2 = x1ax2.reshape(B, C, -1) 
            x1b = x1b.reshape(B, C, -1) 
            x = torch.cat([x1ax2, x1b], dim=2)
        else:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] // shuffle_steps, shuffle_steps)
            x = x.transpose(-1, -2).reshape(x.shape[0], x.shape[1], -1)
        return x

    def unshuffle(self, x):
        B = x.shape[0]
        C = x.shape[1]
        shuffle_steps = self.shuffle_steps
        quotient = x.shape[2] // shuffle_steps
        remainder = x.shape[2] % shuffle_steps
        
        if quotient == 0:
            return x
        if x.shape[2] % self.shuffle_steps != 0:
            x1 = x[..., :remainder * (quotient + 1)]
            x2 = x[..., remainder * (quotient + 1):]
            
            x1 = x1.reshape(B, C, -1, quotient + 1)
            x1 = x1.transpose(-1, -2)
            x1a = x1[:, :, :-1, :]
            x1b = x1[:, :, -1:, :]
            x2 = x2.reshape(B, C, -1, quotient).transpose(-1, -2)
            x1ax2 = torch.cat([x1a, x2], dim=-1)
            x1ax2 = x1ax2.reshape(B, C, -1)
            x1b = x1b.reshape(B, C, -1)
            x = torch.cat([x1ax2, x1b], dim=-1)
            
        else:
            # Shuffle [B, C, L] --> [B, C, shuffle_steps, L//shuffle_steps] --> [B, C, L//shuffle_steps, shuffle_steps] --> [B, C, L]
            x = x.reshape(x.shape[0], x.shape[1], self.shuffle_steps, x.shape[2] // self.shuffle_steps)
            x = x.transpose(-1, -2).reshape(x.shape[0], x.shape[1], -1)
        return x

    def forward(self, x):
        """_summary_

        Args:
            x (torch.tensor): input of size (B, C, L)

        Returns:
            _type_: output of size (B, C, L)
        """
        x = self.shuffle(x)
        if self.fuse_original:
            x = (self.pool(x) + x) / 2
        else:
            x = self.pool(x)
        x = self.unshuffle(x)
        if self.norm:
            return self.norm(x)
        return x

class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        additional_info=None,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        self.additional_info = additional_info
        
        
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None
        
        if (self.additional_info is not None) and \
            self.additional_info["replace_attn"] is not None:
            if self.additional_info['replace_attn']["layer"] == "value":
                self.new_layers = ValueReplace(channels // num_heads)
            elif self.additional_info['replace_attn']["layer"] == "pool":
                self.new_layers = PoolReplace(channels=channels, **self.additional_info['replace_attn']["kwargs"])
            elif self.additional_info['replace_attn']["layer"] == "shuffle_pool":
                self.new_layers = ShufflePoolReplace(channels=channels, **self.additional_info['replace_attn']["kwargs"])
            else:
                raise "not implemented"

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def process_weighted_merging(self, q, k, v, entropy, p ,low_r):
        B, H, K, D = q.shape
        num_l_patch = math.ceil(B  - B * p) - 1 
        e_order = torch.argsort(entropy.squeeze(),  descending=False)
        high = e_order[:-num_l_patch]
        low = e_order[-num_l_patch:]
        h_q, h_k, h_v = q[high], k[high], v[high]
        l_v = v[low]
        e_inverse = torch.argsort(torch.cat([high, low]))
        h_merge, h_unmerge = patch_based_matching(
            h_v, r=int(K*self.additional_info["r"]),
            stride=self.additional_info['stride'],
            no_rand=self.additional_info["no_rand"])

        l_merge, l_unmerge = patch_based_matching(
            l_v, r=K - low_r, 
            stride=K//low_r,
            no_rand=self.additional_info["no_rand"])
        # l_merge, l_unmerge = pool_reduction_sampling(
        #    x0=l_v, kernel_size=128//low_r,
        # )
        
        h_size = torch.ones((h_v.shape[0], H, K, 1), device=v.device)
        l_size = torch.ones((l_v.shape[0], H, K, 1), device=v.device)
        h_size = h_merge(h_size, mode='sum')
        # l_size = l_merge(l_size, mode='sum')
        h_q = h_merge(h_q)
        h_k = h_merge(h_k)
        h_v = h_merge(h_v)
        # l_q = l_merge(l_q)
        # l_k = l_merge(l_k)

        output = {
            'high_q': h_q,
            'high_k': h_k,
            'high_v': h_v,
            'low_q': None,
            'low_k': None,
            'low_v': l_v,
            'high_size': h_size,
            'high_merge': h_merge,
            'high_unmerge': h_unmerge,
            'low_size': l_size,
            'low_merge': l_merge,
            'low_unmerge': l_unmerge,
            'e_inverse':e_inverse
        }
        return output 

        
    def cal_score(self, q:torch.Tensor, k:torch.Tensor, threshold=0.9):   
        q = F.avg_pool1d(q.mean(1).transpose(-1, -2), kernel_size=64, stride=64).transpose(-1, -2).unsqueeze(1).expand(q.shape[0], q.shape[1], -1, q.shape[-1]) 
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        sim = q @ k.mean(-2, keepdim=True).transpose(-1, -2) 
        score =  sim.squeeze().mean(-1).mean(-1)  #[B]
        p = torch.where(score < threshold, 1.0, 0.0).sum(-1)/score.shape[0]
        return  score, p 
    
    def weighted_forward(self, point):
        self.patch_size = min(
            offset2bincount(point['offset']).min().tolist(), self.patch_size_max
        )

        H = self.num_heads
        K = self.patch_size
        C = self.channels
        
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point['serialized_order'][self.order_index][pad]
        inverse = unpad[point['serialized_inverse'][self.order_index]]


        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point['feat'])[order]
        q, k, v = (
            qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
        )
        # attn
        if self.upcast_attention:
            q = q.float()
            k = k.float()
        
        # print(q.shape)

        if q.shape[0] > 100:
            entropy, p = self.cal_score(v, v, 0.4)
            if p >= 0.9:
                q, k, v, size, _, unmerge = self.process_merging(q=q, k=k, v=v, order=order, inverse=inverse)
                feat =  self.self_attn(
                    q=q,
                    k=k,
                    v=v,
                    size=size,
                    unmerge=unmerge
                )
            else:
                output = self.process_weighted_merging(q=q, k=k, v=v, entropy=entropy, p=p, low_r=self.additional_info['low_r'])
                h_feat =  self.self_attn(
                    q=output['high_q'],
                    k=output['high_k'],
                    v=output['high_v'],
                    size=output['high_size'],
                    unmerge=output['high_unmerge'],
                )
                l_feat =  self.self_attn(
                    q=output['low_q'],
                    k=output['low_k'],
                    v=output['low_v'],
                    size=output['low_size'],
                    unmerge=output['low_unmerge'],
                )
                feat = torch.cat([h_feat, l_feat], dim=0)
                feat = feat[output['e_inverse']]
        else:
            q, k, v, size, _, unmerge = self.process_merging(q=q, k=k, v=v, order=order, inverse=inverse)
            feat =  self.self_attn(
                q=q,
                k=k,
                v=v,
                size=size,
                unmerge=unmerge
            )
            
        feat = feat.transpose(1, 2).reshape(-1, C) # (N * K, C)
        feat = feat[inverse]
        # Projection layer
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point['feat'] = feat
        return point

        
    def self_attn(self, q, k ,v,  unmerge, size=None):
                        # Original attention mechanism
        H = self.num_heads
        K = self.patch_size
        C = self.channels
        if v.shape[-2] > 1:
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if size is not None:
                attn = attn + size.log()
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(v.dtype) 
            feat = (attn @ v) # (N', H, K, C // H)
        else:
            feat = v
            print(feat.shape)

        feat = self.process_unreduction(feat, unmerge) # (N, H, K, C // H)
        return feat

    def token_merge_method(self, v, r):
        if self.additional_info["tome"] == 'patch' or  self.additional_info["tome"] == 'weighted_patch':
            merge, unmerge = patch_based_matching(v, r=r, 
                                    stride=self.additional_info["stride"])
        elif self.additional_info["tome"] == 'progressive':
            merge, unmerge = progressive_merging(v, r=r, 
                                threshold=self.additional_info["threshold"])
        elif self.additional_info["tome"] == 'tome':
            merge, unmerge = bipartite_soft_matching(v, r=r)
        elif self.additional_info["tome"] == "base":
            merge, unmerge = bipartite_soft_matching(v, r=0.0)
        elif self.additional_info["tome"] == 'pool':
            merge, unmerge = pool_reduction_sampling(v, self.additional_info["kernel_size"])
        elif self.additional_info["tome"] == 'important_patch':
            merge, unmerge = important_patch_based_matching(v, r=r, 
                                    margin=self.additional_info["margin"],
                                    alpha=self.additional_info["alpha"],
                                    stride=self.additional_info["stride"])     
        elif self.additional_info["tome"] == 'random_patch':
            merge, unmerge = random_patch_based_matching(v, r=r, stride=self.additional_info["stride"])                     
    
        return merge, unmerge
    
    def process_merging(self, q, k, v, order, inverse):
        H = self.num_heads
        K = self.patch_size
        C = self.channels
        size = None
        # self.additional_info["patch_tome"] = True
        if (self.additional_info is not None) and \
                        self.additional_info["tome"] in VALID_TOME_MODES:
            # print("Initialize patch_tome")

            merge, unmerge = self.token_merge_method(v, r=int(K * self.additional_info["r"]))
            
            # Used for cache token merge and unmerge for later use in MLP
            if self.additional_info["trace_back"]:
                self.additional_info["cache_token_merge_unmerge"] = (merge, unmerge)
                self.additional_info["cache_point_order_inverse"] = (order, inverse)
                self.additional_info["patchify_info"] = {
                    "n_segments": q.shape[0],
                    "K": self.patch_size,
                    "H": self.num_heads,
                    "C": self.channels,
                }
            
            if self.additional_info["tome_attention"]:                   
                size = torch.ones((v.shape[0], H, K, 1), device=v.device)
                size = merge(size, mode='sum')
                v = merge(v)
                q = merge(q)
                k = merge(k)
                
        return q, k, v, size, merge, unmerge

    def process_att_replacement(self, v):
        H = self.num_heads
        K = self.patch_size
        C = self.channels
        # replace attn with value
        if (self.additional_info is not None) and \
            self.additional_info["replace_attn"] is not None:
            if self.additional_info["replace_attn"]["layer"] == "value":
                feat = self.new_layers(v)
            elif self.additional_info["replace_attn"]["layer"] == "pool":
                n_segments = v.shape[0]
                feat = v.transpose(1,2).reshape(n_segments, K, C).transpose(1,2)
                feat = self.new_layers(feat).transpose(1,2)
                feat = feat.reshape(n_segments, -1, H, C // H).transpose(1,2)
            elif self.additional_info["replace_attn"]["layer"] == "shuffle_pool":
                n_segments = v.shape[0]
                feat = v.transpose(1,2).reshape(n_segments, K, C).transpose(1,2)
                feat = self.new_layers(feat).transpose(1,2)
                feat = feat.reshape(n_segments, -1, H, C // H).transpose(1,2)
                
        return feat

    def process_unreduction(self, feat, unmerge):     
        return unmerge(feat)
    
    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        if self.additional_info is not None and "copy_point" in self.additional_info.keys():
            from copy import deepcopy
            if self.additional_info["copy_point"]:
                self.copy_point = deepcopy(point)
        
        if self.additional_info is not None and self.additional_info['tome'] == 'weighted_patch' :
            return self.weighted_forward(point)
        
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]
        

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            
            # self.additional_info["patch_tome"] = True
            if (self.additional_info is not None) and \
                            self.additional_info["tome"] in VALID_TOME_MODES:
                # print("Initialize patch_tome")
                q, k, v, size, merge, unmerge = self.process_merging(q, k, v, order, inverse)
                    
            if (self.additional_info is not None) and \
                self.additional_info["replace_attn"] is not None:
                # replace attn with predefined layers
                return self.process_att_replacement(v)
            else:
                # Original attention mechanism
                attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
                print(attn.shape)

                if (self.additional_info is not None) and \
                        self.additional_info["tome"] in VALID_TOME_MODES: 
                    attn = attn + size.log()

                if self.enable_rpe:
                    rpe_feats = self.rpe(self.get_rel_pos(point, order))
                    if (self.additional_info is not None) and \
                        self.additional_info["tome"] in VALID_TOME_MODES:
                        rpe_feats = merge(rpe_feats)
                        rpe_feats = rpe_feats.transpose(-2,-1)
                        rpe_feats = merge(rpe_feats)
                        rpe_feats = rpe_feats.transpose(-2,-1)
                        attn = attn + rpe_feats
                if self.upcast_softmax:
                    attn = attn.float()
                attn = self.softmax(attn)
                attn = self.attn_drop(attn).to(qkv.dtype) 
                feat = (attn @ v) # (N', H, K, C // H)

            # unmerge and reshape feat: (N', H, K, C // H) => (N', K, H, C // H)
            if (self.additional_info is not None) and \
                (self.additional_info["tome"] in VALID_TOME_MODES) and \
                    self.additional_info["tome_attention"]:

                feat = self.process_unreduction(feat, unmerge) # (N, H, K, C // H)
                
                
            feat = feat.transpose(1, 2).reshape(-1, C) # (N * K, C)
        else:
            # Using flash attention
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        
        feat = feat[inverse]

        # Projection layer
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
        additional_info=None,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.additional_info = additional_info
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward_v2(self, x):        
        merge, unmerge = self.additional_info["cache_token_merge_unmerge"]
        order, inverse = self.additional_info["cache_point_order_inverse"]
        # encode and reshape x: (N', K, H, C') => (N', H, K, C')
        x = x[order]
        n_segments = self.additional_info["patchify_info"]["n_segments"]
        H = self.additional_info["patchify_info"]["H"]
        K = self.additional_info["patchify_info"]["K"]
        C = self.additional_info["patchify_info"]["C"]
        x = x.reshape(n_segments, K, H, C // H).permute(0, 2, 1, 3)
        if self.additional_info["single_head_tome"]:
            x =  x.transpose(1,2).reshape(n_segments, K, C)[:,None,:,:]
        x = merge(x)
        if self.additional_info["single_head_tome"]:
            x =  x.reshape(n_segments, -1, H, C // H).transpose(1,2)
        # Reshape back for FC layer: (N', H, K', C') => (N', K', H, C') => (-1, H * C')
        x = x.transpose(1, 2).reshape(-1, C) # (N' * K', C)
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # Reshape back for Unmerge layer: (N' * K', C) => (N', K', H, C') => (N', H, K', C') 
        x = x.reshape(n_segments, -1, H, C // H).transpose(1, 2)
        if self.additional_info["single_head_tome"]:
            x =  x.transpose(1,2).reshape(n_segments, -1, C)[:,None,:,:]
        x = unmerge(x)
        if self.additional_info["single_head_tome"]:
            x =  x.reshape(n_segments, K, H, C // H).transpose(1,2)
        x = x.transpose(1, 2).reshape(-1, C)
        x = x[inverse]
        return x

    def forward(self, x):
        if self.additional_info is not None:
            if self.additional_info["trace_back"] and self.additional_info["tome_mlp"]:
                return self.forward_v2(x)
        # print("MLP")
        # breakpoint()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        additional_info=None,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.additional_info = additional_info

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            additional_info=additional_info,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
                additional_info=additional_info,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"
        # breakpoint()

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


@MODELS.register_module("PT-v3m1")
class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        additional_info=None,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        additional_info=additional_info,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            additional_info=additional_info,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        if not self.cls_mode:
            point = self.dec(point)
        # else:
        #     point.feat = torch_scatter.segment_csr(
        #         src=point.feat,
        #         indptr=nn.functional.pad(point.offset, (1, 0)),
        #         reduce="mean",
        #     )
        return point