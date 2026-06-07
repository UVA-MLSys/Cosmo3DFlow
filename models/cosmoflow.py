import os
import torch
import torch.utils.checkpoint
import torch.nn as nn
from torch_ema import ExponentialMovingAverage
from typing import Any, Dict, Tuple, Union

import torch
from torch import nn
from lightning.pytorch import utilities
from lightning import LightningModule

from torchdyn.core import NeuralODE

import torch.nn.functional as F

from models.DWT_IDWT_layer import DWT_3D, IDWT_3D
from utils.utils import compute_power_spectrum_loss
from utils.visualize import *
initialize_plt()

# %%
from models import layers, layerspp
import functools

ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act

default_initializer = layers.default_init

from einops import rearrange

import torch
import torch.nn as nn
import math

class KWeightedPowerSpectrumLoss3D(nn.Module):
    """
    Computes a frequency-weighted Power Spectrum loss between two 3D volumes.
    """
    def __init__(self, grid_size: int, box_size: float = 1000.0, decay_rate: float = 2.0):
        super().__init__()
        self.grid_size = grid_size
        self.box_size = box_size
        
        # 1. Calculate physical grid spacing (dx)
        dx = box_size / grid_size
        
        # 2. Create a 3D grid of physical frequencies (k = 2 * pi * f)
        # fftfreq(d=dx) gives cycles per Mpc/h. Multiply by 2*pi for angular wavenumber k
        k_1d = torch.fft.fftfreq(grid_size, d=dx) * 2 * math.pi
        kx, ky, kz = torch.meshgrid(k_1d, k_1d, k_1d, indexing='ij')
        
        # Calculate the magnitude of the wavevector |k| 
        k_mag = torch.sqrt(kx**2 + ky**2 + kz**2)
        
        # 3. Physical 1D Nyquist limit: k_nyq = pi / dx
        k_nyquist_1d = math.pi / dx
        
        # Normalize by the exact physical 1D Nyquist limit
        k_mag_norm = k_mag / k_nyquist_1d
        
        # Exponential decay
        weight_mask = torch.exp(-decay_rate * k_mag_norm)
        
        # HARD CUTOFF: Zero out the anisotropic grid artifacts (k > k_nyquist_1d)
        isotropic_mask = (k_mag <= k_nyquist_1d).float()
        weight_mask = weight_mask * isotropic_mask
        
        weight_mask[0, 0, 0] = 0.0 # Ignore DC component (k=0)
        weight_mask = weight_mask.view(1, 1, grid_size, grid_size, grid_size)
        
        self.register_buffer('weight_mask', weight_mask, persistent=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: [B, C, D, H, W]
        """
        # cuFFT frequently crashes with fp16/bf16 inputs during AMP training
        pred = pred.to(torch.float32)
        target = target.to(torch.float32)
        # Fast Fourier Transform
        fft_pred = torch.fft.fftn(pred, dim=(-3, -2, -1), norm='ortho')
        fft_target = torch.fft.fftn(target, dim=(-3, -2, -1), norm='ortho')
        
        # Power Spectrum
        ps_pred = torch.abs(fft_pred) ** 2
        ps_target = torch.abs(fft_target) ** 2
        
        log_ps_pred = torch.log(ps_pred + 1e-8)
        log_ps_target = torch.log(ps_target + 1e-8)
        
        # self.weight_mask will automatically be on the same device as pred/target
        # assuming you initialized this module inside your LightningModule or called .to(device)
        weighted_squared_error = self.weight_mask * (log_ps_pred - log_ps_target) ** 2
        
        return torch.mean(weighted_squared_error)

class WaveletAdaptiveNorm(nn.Module):
    """Normalization that treats different wavelet scales separately"""
    def __init__(self, num_channels, num_wavelet_bands=8):
        super().__init__()
        self.num_bands = num_wavelet_bands
        
        # Ensure num_channels is divisible by num_bands
        assert num_channels % num_wavelet_bands == 0, \
            f"num_channels ({num_channels}) must be divisible by num_wavelet_bands ({num_wavelet_bands})"
        
        self.channels_per_band = num_channels // num_wavelet_bands
        
        # Separate normalization per wavelet band
        self.norms = nn.ModuleList([
            nn.GroupNorm(
                num_groups=min(self.channels_per_band // 4, 8),
                num_channels=self.channels_per_band,
                eps=1e-6
            )
            for _ in range(num_wavelet_bands)
        ])
    
    def forward(self, x):
        # Split by wavelet bands
        bands = torch.chunk(x, self.num_bands, dim=1)
        
        # Normalize each band separately
        normalized = [norm(band) for norm, band in zip(self.norms, bands)]
        
        return torch.cat(normalized, dim=1)

# %%
class UNet3DModel(nn.Module):

  def __init__(self, config, 
               use_scale_conditioning=False,
               use_cross_scale_skips=False,
               use_wavelet_adaptive_norm=False,
               use_checkpoint=False):
    super().__init__()
    self.config = config
    self.act = act = get_act(config)
    
    # Novel feature flags
    self.use_scale_conditioning = use_scale_conditioning
    self.use_cross_scale_skips = use_cross_scale_skips
    self.use_wavelet_adaptive_norm = use_wavelet_adaptive_norm
    self.use_checkpoint = use_checkpoint

    self.nf = nf = config.model.nf
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    dropout = config.model.dropout
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = [config.model.image_size // (2 ** i) for i in range(num_resolutions)]

    self.conditional = conditional = config.model.conditional  # noise-conditional
    fir = config.model.fir
    fir_kernel = config.model.fir_kernel
    self.skip_rescale = skip_rescale = config.model.skip_rescale

    self.embedding_type = embedding_type = config.model.embedding_type.lower()
    init_scale = config.model.init_scale
    assert embedding_type in ['fourier']

    modules = []
    # timestep/noise_level embedding; only for continuous training

    modules.append(layerspp.GaussianFourierProjection(
      embedding_size=nf, scale=config.model.fourier_scale
    ))
    embed_dim = 2 * nf

    if conditional:
      modules.append(nn.Linear(embed_dim, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)

    ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                    act=act,
                                    dropout=dropout,
                                    fir=fir,
                                    fir_kernel=fir_kernel,
                                    init_scale=init_scale,
                                    skip_rescale=skip_rescale,
                                    temb_dim=nf * 4)

    # Downsampling block
    input_channels = config.model.num_input_channels
    output_channels = config.model.num_output_channels

    # NEW: Scale-specific conditioning projections
    if self.use_scale_conditioning:
      self.scale_projections = nn.ModuleList([
        conv1x1(8, nf * ch_mult[i])  # 8 wavelet bands -> features
        for i in range(num_resolutions)
      ])
    
    # NEW: Cross-scale skip connections
    if self.use_cross_scale_skips:
      self.cross_scale_skips = nn.ModuleList([
        conv1x1(nf * ch_mult[i], nf * ch_mult[i])
        for i in range(num_resolutions)
      ])

    # Downsampling block
    modules.append(conv3x3(input_channels, nf))
    hs_c = [nf]

    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch
        hs_c.append(in_ch)

      if i_level != num_resolutions - 1:
        modules.append(ResnetBlock(down=True, in_ch=in_ch))
        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                   out_ch=out_ch))
        in_ch = out_ch

      if i_level != 0:
        modules.append(ResnetBlock(in_ch=in_ch, up=True))

    assert not hs_c

    # NEW: Wavelet-adaptive normalization vs standard GroupNorm
    if self.use_wavelet_adaptive_norm:
      modules.append(WaveletAdaptiveNorm(in_ch, num_wavelet_bands=8))
    else:
      modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                  num_channels=in_ch, eps=1e-6))
    
    modules.append(conv3x3(in_ch, output_channels, init_scale=init_scale))

    self.all_modules = nn.ModuleList(modules)


  def forward(self, x, t, z):
    # Store z separately for scale conditioning
    z_original = z if self.use_scale_conditioning else None
    
    # Standard concatenation for baseline
    x = torch.concat([x, z], dim=1)
    
    if t.ndim == 0:
      t = t.unsqueeze(0).expand(x.shape[0])
    
    def run_module(module, *args):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    modules = self.all_modules
    m_idx = 0
    if self.embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      used_sigmas = t
      temb = modules[m_idx](used_sigmas)
      m_idx += 1

    if self.conditional:
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None

    # Downsampling block
    hs = [modules[m_idx](x)]
    m_idx += 1
    
    # NEW: Store features at each scale for cross-scale skips
    scale_features = {} if self.use_cross_scale_skips else None
    
    for i_level in range(self.num_resolutions):
      # NEW: Scale-specific conditioning
      if self.use_scale_conditioning and z_original is not None:
        # Project wavelet bands to feature space
        z_features = self.scale_projections[i_level](z_original)
        
        # Downsample to match current resolution
        if i_level > 0:
          z_features = F.avg_pool3d(z_features, kernel_size=2**i_level, stride=2**i_level)
      
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = run_module(modules[m_idx], hs[-1], temb)
        
        # NEW: Inject scale-specific conditioning
        if self.use_scale_conditioning and z_original is not None:
          if h.shape == z_features.shape:
            h = h + z_features
          else:
            # Adaptive pooling if shapes don't match
            z_adapted = F.adaptive_avg_pool3d(z_features, h.shape[-3:])
            h = h + z_adapted
        
        m_idx += 1
        hs.append(h)

      if i_level != self.num_resolutions - 1:
        h = run_module(modules[m_idx], hs[-1], temb)
        m_idx += 1
        hs.append(h)
      
      #TODO: this part is wrong and needs to be fixed. But that needs retraining all 
      # current checkpoints, so will do later.
      # NEW: Store features for cross-scale skips
      if self.use_cross_scale_skips:
        scale_features[i_level] = hs[-1].clone()

    h = hs[-1]
    h = run_module(modules[m_idx], h, temb)
    m_idx += 1

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = run_module(modules[m_idx], torch.cat([h, hs.pop()], dim=1), temb)
        m_idx += 1

      # NEW: Add cross-scale residual connection
      if self.use_cross_scale_skips and i_level in scale_features:
        skip_h = scale_features[i_level]
        if skip_h.shape == h.shape:
          cross_scale = self.cross_scale_skips[i_level](skip_h)
          h = h + cross_scale

      if i_level != 0:
        h = run_module(modules[m_idx], h, temb)
        m_idx += 1

    assert not hs

    h = self.act(modules[m_idx](h))
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1

    assert m_idx == len(modules)

    return h

# %% [markdown]
# ## FlowMatching

# %%
class AdaptiveWaveletMixer(nn.Module):
    def __init__(self, num_bands=8, hidden_dim=64):
        super().__init__()
        self.num_bands = num_bands
        
        # Learn scale-dependent flow speeds as function of time
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_bands),
            nn.Softplus()  # Ensure positive speeds
        )
    
    def forward(self, x_wavelet, t):
        """
        x_wavelet: (B, 8, H, W, D) - wavelet coefficients
        t: (B,) - time
        Returns: (B, 8, H, W, D) with time-modulated scales
        """
        # Get scale-dependent speeds
        speeds = self.time_encoder(t.view(-1, 1))  # (B, 8)
        speeds = speeds.view(-1, 8, 1, 1, 1)
        
        # Modulate each scale by its speed
        return x_wavelet * speeds

# %%
class ConditionedVelocityModel(nn.Module):
    """Neural net for velocity field prediction, with optional reverse flag"""

    def __init__(
        self,
        velocity_model: torch.nn.Module,
        h: torch.Tensor | None,
        reverse: bool = False,
    ):
        super(ConditionedVelocityModel, self).__init__()
        self.reverse = reverse
        self.velocity_model = velocity_model
        self.h = h

    def forward(
        self,
        t: torch.Tensor | int,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if h is None:
            h = self.h
        velocity = self.velocity_model(x, t=t, z=h)
        return -velocity if self.reverse else velocity

class FlowMatching(nn.Module):
    """Flow matching module with training loss and inference-time log-likelihood."""

    def __init__(
        self,
        velocity_model: torch.nn.Module,
        grid_size: int,
        sigma: float = 0.0,
        reverse: bool = False,
        use_wavelet: bool = True,
        power_spec_weight = 0.0,
        wavelet_type: str = 'haar'
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.sigma = sigma
        self.reverse = reverse
        self.dwt = DWT_3D(wavelet_type) # bior4.4, haar
        self.idwt = IDWT_3D(wavelet_type)
        
        self.use_wavelet = use_wavelet
        self.power_spec_weight = power_spec_weight

        # self.ps_loss_fn = KWeightedPowerSpectrumLoss3D(
        #     grid_size=grid_size, decay_rate=2.0
        # )


    def get_mu_t(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        '''Sample distribution mean for rectified flow matching'''
        return t * x1 + (1 - t) * x0

    def get_gamma_t(self, t: torch.Tensor) -> torch.Tensor:
        '''Sample distribution variance for rectified flow matching'''
        return torch.sqrt(2 * t * (1 - t))

    def sample_xt(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, eps: torch.Tensor | None
    ) -> torch.Tensor:
        '''Sample from distribution at time t'''
        t_broadcast = t.view(t.shape[0], *([1] * (x0.dim() - 1)))
        mu_t = self.get_mu_t(x0, x1, t_broadcast)
        if self.sigma != 0.0:
            sigma_t = self.get_gamma_t(t_broadcast)
            return mu_t + sigma_t * eps
        return mu_t
    
    def compute_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        h: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if self.use_wavelet:
            return self.compute_wavelet_loss(x0, x1, h, t)
        else:
            return self.compute_pixel_loss(x0, x1, h, t)

    def forward(
        self, x0: torch.Tensor,
        h: torch.Tensor | None = None,
        n_sampling_steps: int = 50,
        solver: str = "euler",
        full_return: bool = False
    ):
        return self.predict(x0, h, n_sampling_steps, solver, full_return)
        
    def predict(
        self,
        x0: torch.Tensor,
        h: torch.Tensor | None = None,
        n_sampling_steps: int = 50,
        solver: str = "euler",
        full_return: bool = False,
    ) -> torch.Tensor:
        if self.use_wavelet:
            return self.predict_wavelet(x0, h, n_sampling_steps, solver, full_return)
        else:
            return self.predict_pixel(x0, h, n_sampling_steps, solver, full_return)
    
    def compute_pixel_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        h: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        '''Flow matching loss'''
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device).type_as(x0)

        eps = torch.randn_like(x0) if self.sigma != 0.0 else None
        xt = self.sample_xt(x0, x1, t, eps)
        ut = x1 - x0
        vt = self.velocity_model(xt, t=t, z=h)
        
        loss = torch.mean((vt - ut) ** 2)
        
        if self.power_spec_weight > 0:
            # Log-space MSE
            ps_loss = compute_power_spectrum_loss(vt, ut)
            total_loss = loss + self.power_spec_weight * ps_loss
            return {
                'loss': total_loss,
                'recon_loss': loss,
                'ps_loss': ps_loss
            }
        
        return loss

    def predict_pixel(
        self,
        x0: torch.Tensor,
        h: torch.Tensor | None = None,
        n_sampling_steps: int = 50,
        solver: str = "euler",
        full_return: bool = False,
    ) -> torch.Tensor:
        '''Run inference by solving probability flow ODE with initial condition x0'''
        conditional_velocity_model = ConditionedVelocityModel(
            velocity_model=self.velocity_model, h=h, reverse=self.reverse
        )
        node = NeuralODE(
            conditional_velocity_model,
            solver=solver,
            sensitivity="adjoint",
        )
        with torch.no_grad():
            traj = node.trajectory(
                x0,
                t_span=torch.linspace(0, 1, n_sampling_steps, device=x0.device),
            )
        return traj[-1] if not full_return else traj
    
    def dwt_(self, x):
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.dwt(x)
        return torch.cat([LLL / (8**0.5), LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
    
    def idwt_(self, x):
        B, _, H, W, D = x.size()
        return self.idwt(
            # Must use reshape().contiguous() as slicing channel dim breaks memory contiguity
            x[:, 0, :, :, :].reshape(B, 1, H, W, D).contiguous() * (8**0.5),
            x[:, 1, :, :, :].reshape(B, 1, H, W, D).contiguous(),
            x[:, 2, :, :, :].reshape(B, 1, H, W, D).contiguous(),
            x[:, 3, :, :, :].reshape(B, 1, H, W, D).contiguous(),
            x[:, 4, :, :, :].reshape(B, 1, H, W, D).contiguous(),
            x[:, 5, :, :, :].reshape(B, 1, H, W, D).contiguous(),
            x[:, 6, :, :, :].reshape(B, 1, H, W, D).contiguous(),
            x[:, 7, :, :, :].reshape(B, 1, H, W, D).contiguous()
        )

    def compute_wavelet_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        h: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        '''Flow matching loss'''
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device).type_as(x0)
            
        eps = torch.randn_like(x0) if self.sigma != 0.0 else None
        if eps is not None:
            eps = self.dwt_(eps)
            
        x0 = self.dwt_(x0)
        x1 = self.dwt_(x1)
        
        xt = self.sample_xt(x0, x1, t, eps)
        ut = x1 - x0
        
        h = self.dwt_(h)
        
        vt = self.velocity_model(xt, t=t, z=h)
        
        loss = torch.mean((vt - ut) ** 2)
        
        if self.power_spec_weight>0:
            ps_loss = compute_power_spectrum_loss(
                self.idwt_(vt), self.idwt_(ut)
            )
            total_loss = loss + self.power_spec_weight * ps_loss
            return {
                'loss': total_loss,
                'recon_loss': loss,
                'ps_loss': ps_loss
            }
        
        return loss

    def predict_wavelet(
        self,
        x0: torch.Tensor,
        h: torch.Tensor | None = None,
        n_sampling_steps: int = 50,
        solver: str = "euler",
        full_return: bool = False,
    ) -> torch.Tensor:
        '''Run inference by solving probability flow ODE with initial condition x0'''
        x0 = self.dwt_(x0)
        h = self.dwt_(h)
        
        conditional_velocity_model = ConditionedVelocityModel(
            velocity_model=self.velocity_model, h=h, reverse=self.reverse
        )
        node = NeuralODE(
            conditional_velocity_model,
            solver=solver,
            sensitivity="adjoint",
        )
        with torch.no_grad():
            traj = node.trajectory(
                x0,
                t_span=torch.linspace(0, 1, n_sampling_steps, device=x0.device),
            )
            
        return self.idwt_(traj[-1]) if not full_return else [self.idwt_(tr) for tr in traj]

# %% [markdown]
# ## CosmoFlow3D

# %%
from torch_ema import ExponentialMovingAverage

class CosmoFlow3D(LightningModule):
    """LightningModule for flow-matching based cosmological field compression in 3D."""

    def __init__(
        self,
        config, unconditional: bool = False,
        reverse: bool = False,
        learning_rate=1e-4,
        use_wavelet=True,
        use_ema=True, 
        power_spec_weight: float = 0.0,
        use_scale_conditioning=False,
        use_cross_scale_skips=False,
        use_wavelet_adaptive_norm=False,
        use_checkpoint=False,
        compile_model=False,
        wavelet_type: str = 'haar'
    ):
        super().__init__()
        self.save_hyperparameters()

        self.unconditional = unconditional
        velocity_model = UNet3DModel(
            config,
            use_scale_conditioning=use_scale_conditioning,
            use_cross_scale_skips=use_cross_scale_skips,
            use_wavelet_adaptive_norm=use_wavelet_adaptive_norm,
            use_checkpoint=use_checkpoint
        )

        if compile_model:
            velocity_model = torch.compile(velocity_model)

        self.decoder = FlowMatching(
            velocity_model,
            grid_size=config.model.image_size,
            reverse=reverse, 
            use_wavelet=use_wavelet,
            power_spec_weight=power_spec_weight,
            wavelet_type=wavelet_type
        )
        
        self.use_ema = use_ema
        if use_ema: self.ema = None
        
    def configure_model(self):
        """Called before load_from_checkpoint loads state_dict"""
        if self.use_ema and self.ema is None:
            self.ema = ExponentialMovingAverage(
                self.decoder.velocity_model.parameters(),
                decay=0.99
            )
            self.ema.to(self.device)
        
    def get_loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x, y = batch
        t = torch.rand((y.shape[0],), device=y.device)
        x0 = torch.randn_like(y)

        return self.decoder.compute_loss(x0=x0, x1=y, h=x, t=t)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        # self.optimizers().step()
        loss = self.get_loss(batch)
        
        if type(loss) == dict:
            loss_dict = {f'train_{key}': loss[key] for key in loss}
            self.log_dict(loss_dict, prog_bar=True, sync_dist=True, on_epoch=True)
            loss = loss["loss"]
        else:
            self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
            
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        t = torch.rand((y.shape[0],), device=y.device)
        x0 = torch.randn_like(y)
        loss = self.decoder.compute_loss(x0=x0, x1=y, h=x, t=t)
        
        if type(loss) == dict:
            loss_dict = {f'val_{key}': loss[key] for key in loss}
            self.log_dict(loss_dict, prog_bar=True, sync_dist=True, on_epoch=True, logger=True)
            loss = loss["loss"]
        else:
            self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True, logger=True)
            
            return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, min_lr=1e-8
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
        
    def on_before_zero_grad(self, *args, **kwargs):
        # Update EMA after optimizer step but before zero_grad
        if self.use_ema:
            self.ema.update()