# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Layers for defining NCSN++.
"""
from models import layers
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
default_init = layers.default_init
naive_upsample = layers.naive_upsample
naive_downsample = layers.naive_downsample

class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    # self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
    # Use register_buffer instead of nn.Parameter for non-trainable tensors
    self.register_buffer('W', torch.randn(embedding_size) * scale)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).contiguous()

class Combine(nn.Module):
  """Combine information from skip connections."""

  def __init__(self, dim1, dim2, method='cat'):
    super().__init__()
    self.Conv_0 = conv1x1(dim1, dim2)
    self.method = method

  def forward(self, x, y):
    h = self.Conv_0(x)
    if self.method == 'cat':
      return torch.cat([h, y], dim=1).contiguous()
    elif self.method == 'sum':
      return h + y
    else:
      raise ValueError(f'Method {self.method} not recognized.')


class ResnetBlockBigGANpp(nn.Module):
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False,
               dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
               skip_rescale=True, init_scale=0., use_scale_shift_norm=False):
    super().__init__()

    out_ch = out_ch if out_ch else in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
    self.up = up
    self.down = down
    self.use_scale_shift_norm = use_scale_shift_norm
    self.fir = fir
    self.fir_kernel = fir_kernel
    self.Conv_0 = conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch * 2 if use_scale_shift_norm else out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
      nn.init.zeros_(self.Dense_0.bias)

    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch or up or down:
      self.Conv_2 = conv1x1(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.in_ch = in_ch
    self.out_ch = out_ch

  def forward(self, x, temb=None):
    h = self.act(self.GroupNorm_0(x.contiguous()))

    if self.up:
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        x = F.interpolate(x, scale_factor=2, mode='nearest')
    elif self.down:
        h = F.avg_pool3d(h, kernel_size=2)
        x = F.avg_pool3d(x, kernel_size=2)


    h = self.Conv_0(h.contiguous())
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      emb_out = self.Dense_0(self.act(temb))[:, :, None, None, None]
      if self.use_scale_shift_norm:
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.GroupNorm_1(h)
        h = h * (1 + scale) + shift
        h = self.act(h)
      else:
        h = h + emb_out
        h = self.act(self.GroupNorm_1(h))
    else:
      h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h.contiguous())

    if self.in_ch != self.out_ch or self.up or self.down:
      x = self.Conv_2(x.contiguous())
    h = h.contiguous()
    
    if not self.skip_rescale: # This is where non-contiguous x would be added
      return x + h
    else:
      return (x + h) / np.sqrt(2.)
