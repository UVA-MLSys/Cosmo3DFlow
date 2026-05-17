import torch
import numpy as np
import json
from torch import nn

def init_weights(m):
  if isinstance(m, (nn.Linear, nn.Conv3d)): # Add Conv3d
      nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_in", nonlinearity="relu")
      if m.bias is not None:
          m.bias.data.fill_(0.0)


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def dict_to_config(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_config(v)
    return Config(**d)


def get_config(config_path):
    # Load parameters from a json file back into the Config class
    with open(config_path, 'r') as f:
        loaded_config_dict = json.load(f)

    # Convert dictionaries back into Config objects
    loaded_config = dict_to_config(loaded_config_dict)

    return loaded_config


# Define sigma(t) mapping
def get_sigma_time(sigma_min, sigma_max):
    def sigma_time(t):
        return sigma_min * (sigma_max / sigma_min) ** t
    return sigma_time

# Define time uniform sampling
def get_sample_time(sampling_eps, T):
    def sample_time(shape, device='cpu'):
        return (sampling_eps - T) * torch.rand(shape, device=device) + T
    return sample_time



class VESDE():
  def __init__(self, sigma_min, sigma_max, N, T = 1, eps=1e-5, device='cpu'):
    super().__init__()
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.N = N
    self.T = T
    self.eps = eps
    self.device = device

    self.timesteps = torch.linspace(T, eps, N, device=self.device)

  def prior_sampling(self, shape, device='cpu'):
    return torch.randn(*shape, device=device) * self.sigma_max

  def sample_time(self, shape, device=None):
    device = device if device is not None else self.device
    return (self.eps - self.T) * torch.rand(shape, device=device) + self.T

  def sigma_fn(self, t):
    return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

  def sde(self, x, t):
    sigma = self.sigma_fn(t)
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    return drift, diffusion

  def rsde(self, x, t, model_output):
    """Create the drift and diffusion functions for the reverse SDE/ODE."""
    drift, diffusion = self.sde(x, t)
    score = self.score_fn(t, model_output)
    drift = drift - diffusion[:, None, None, None, None] ** 2 * score
    return drift, diffusion

  def score_fn(self, t, model_output):
    return model_output/self.sigma_fn(t)[:,None,None,None,None]

  def update_fn(self, x, t, model_output):
    dt = -self.T / self.N
    z = torch.randn_like(x)
    drift, diffusion = self.rsde(x, t, model_output)
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None, None, None, None] * np.sqrt(-dt) * z
    return x, x_mean

def compute_power_spectrum_3d(field):
    """Fast GPU power spectrum up to Nyquist limit"""
    # field shape: (B, 1, H, W, D) or (B, H, W, D)
    if field.ndim == 5:
        field = field.squeeze(1)  # (B, H, W, D)
    
    B, H, W, D = field.shape
    
    # FFT (on GPU)
    fft_field = torch.fft.rfftn(field, dim=(-3, -2, -1))
    power = torch.abs(fft_field) ** 2
    
    # k-space grid (in units of fundamental mode)
    kx = torch.fft.fftfreq(H, d=1.0, device=field.device) * H
    ky = torch.fft.fftfreq(W, d=1.0, device=field.device) * W
    kz = torch.fft.rfftfreq(D, d=1.0, device=field.device) * D
    
    kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = torch.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
    
    # Nyquist limit: k_nyquist = N/2 (for grid of size N)
    k_nyquist = min(H, W, D) / 2.0
    
    # Radial binning up to Nyquist only
    n_bins = 15
    k_edges = torch.linspace(0, k_nyquist, n_bins + 1, device=field.device)
    
    power_spectrum = []
    for i in range(n_bins):
        mask = (k_mag >= k_edges[i]) & (k_mag < k_edges[i+1])
        if mask.sum() > 0:
            # Average power in this bin
            binned = power[:, mask].mean(dim=1)  # (B,)
            power_spectrum.append(binned)
        else:
            power_spectrum.append(torch.zeros(B, device=field.device))
    
    return torch.stack(power_spectrum, dim=1)  # (B, n_bins)

def compute_power_spectrum_loss(field1, field2):
    power1 = compute_power_spectrum_3d(field1)
    power2 = compute_power_spectrum_3d(field2)
    return torch.mean((torch.log(power1 + 1e-8) - torch.log(power2 + 1e-8)) ** 2)


def nbodykit_pspec(x, boxsize=1000.0, kmax=None):
  from nbodykit.lab import ArrayMesh, FFTPower

  mesh = ArrayMesh(x, BoxSize=boxsize)
  result = FFTPower(mesh, mode='1d', kmax=kmax)
  PS = result.power
  return PS['power'].real, PS['k']

def nbodykit_cross_pspec(x, y, boxsize=1000., kmax=None):
  from nbodykit.lab import ArrayMesh, FFTPower

  meshx = ArrayMesh(x, BoxSize=boxsize)
  meshy = ArrayMesh(y, BoxSize=boxsize)
  resultxy = FFTPower(first=meshx, mode='1d', second=meshy, kmax=kmax)
  resultxx = FFTPower(first=meshx, mode='1d', kmax=kmax)
  resultyy = FFTPower(first=meshy, mode='1d', kmax=kmax)

  PS_xy = resultxy.power['power'].real
  PS_xx = resultxx.power['power'].real
  PS_yy = resultyy.power['power'].real
  k = resultxy.power['k']

  # Now do the division
  PS = PS_xy / np.sqrt(PS_xx * PS_yy)

  return PS, k