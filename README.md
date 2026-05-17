# Cosmo3DFlow: Wavelet Flow Matching for Spatial-to-Spectral Compression in Reconstructing the Early Universe

[![KDD 2026](https://img.shields.io/badge/KDD-2026-blue?style=flat-square)](https://kdd.org/kdd2026/)
[![arXiv](https://img.shields.io/badge/arXiv-2602.10172-b31b1b?style=flat-square)](https://arxiv.org/abs/2602.10172)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)

> **KDD '26** · ACM SIGKDD · August 9–13, 2026 · Jeju, Republic of Korea

---

## Overview

Cosmo3DFlow reconstructs early-Universe initial conditions from present-day observations using **3D Wavelet Flow Matching** — operating entirely in wavelet space for a **50× speedup** over diffusion baselines.

- **50×** faster sampling than score-based diffusion (5.2 s vs. 243 s at 128³)
- **8×** spatial compression via single-level 3D Haar DWT
- **10×** fewer ODE steps · **2× less memory** · better reconstruction quality

<p align="center">
  <img src="docs/figures/teaser.png" alt="Cosmo3DFlow teaser" width="800"/>
</p>

---

## The Void Problem

~**63.7%** of cosmic volume is empty voids holding only **16.2%** of dark matter mass — yet voxel-space models spend equal compute everywhere. The 3D DWT converts spatial emptiness into spectral sparsity, concentrating compute on physically meaningful filaments and halos.

<p align="center">
  <em>Left: voxel grid — uniform compute over 1.3 M empty voxels at 128³. Right: wavelet representation — voids collapse to a handful of coarse coefficients; filaments get dense fine-grained detail.</em>
  <br><br>
  <img src="docs/figures/SparsityDualis.png" alt="Voxel vs wavelet representation of the cosmic web" width="560"/>
</p>

---

## Method

### Wavelet Flow Matching

Flow matching trained entirely in wavelet space: apply 3D Haar DWT → interpolate the flow path → train with a flow + power-spectrum loss → integrate 100 Euler steps → IDWT to recover the density field.

### Wavelet-Aware 3D U-Net

<p align="center">
  <em>16-channel input (8ch wavelet noise + 8ch conditioning) → 8-channel velocity field, with scale-specific conditioning and cross-scale skip connections at every resolution.</em>
  <br><br>
  <img src="docs/figures/3dunet.png" alt="Wavelet-aware 3D U-Net architecture" width="600"/>
</p>

- **Scale-specific conditioning** — per-level wavelet features injected via 1×1×1 convolutions
- **Cross-scale skip connections** — encoder features bridged to non-corresponding decoder levels
- **BigGAN residual blocks** — GroupNorm · SiLU · Gaussian Fourier time embeddings
- **Fixed 8³ bottleneck** — 2 / 3 / 4 encoder levels for 32³ / 64³ / 128³

### Training

```
Optimizer: AdamW  lr=1e-4  |  Grad clip: 1.0  |  EMA: 0.999
Schedule:  ReduceLROnPlateau (patience=5, factor=0.5)
Epochs:    100 (best val-loss)  |  Batch: 16 / 8 / 4  (32³/64³/128³)
Hardware:  NVIDIA A100 80 GB
```

---

## Dataset

Three [Quijote N-body](https://quijote-simulations.readthedocs.io/en/latest/LH.html) suites · (1000 h⁻¹ Mpc)³ boxes · 512³ particles · fields at 32³, 64³, 128³

| Suite | Simulations | Split |
|---|---|---|
| Standard Latin Hypercube (LH) | 2,000 | 1800 / 100 / 100 |
| Big Sobol Sequence (BSQ) | 1,000 | 8:1:1 |
| Non-Gaussian fNL LH | 1,000 | 8:1:1 |

---

## Experiments

### Qualitative Results

<p align="center">
  <em><strong>Reconstruction quality.</strong> 2D slices at z = 127 for three Standard LH test samples. Columns: observation (z = 0) · ground truth · Diffusion · Cosmo3DFlow · error maps. The baseline blurs fine structure; Cosmo3DFlow preserves sharp filamentary features.</em>
  <br><br>
  <img src="docs/figures/multiple_samples.png" alt="Qualitative reconstruction comparison" width="800"/>
</p>

### Computational Efficiency

<p align="center">
  <em><strong>Efficiency vs. quality trade-off.</strong> Cosmo3DFlow is 4.4× faster at equal steps and achieves better accuracy with 10× fewer steps — combining for a 50× end-to-end speedup.</em>
  <br><br>
  <img src="docs/figures/efficiency_comparison.png" alt="Efficiency comparison" width="500"/>
</p>

| | Cosmo3DFlow | Diffusion |
|---|---|---|
| Sampling time @ 128³ | **5.2 s** | 243 s |
| Peak memory @ 128³ | **2.1 GB** | 4.0 GB |
| ODE steps | **100** | 1,000 |

### Convergence

<p align="center">
  <em><strong>Convergence vs. integration steps.</strong> Cosmo3DFlow reaches its best VRMSE in 100 steps; diffusion requires 1,000 to reach a higher error floor.</em>
  <br><br>
  <img src="docs/figures/convergence_steps.png" alt="Convergence vs ODE steps" width="800"/>
</p>

### Physics Validation

<p align="center">
  <em><strong>Physics metrics.</strong> Power spectrum P(k), cross-correlation C(k), and transfer function T(k) vs. wavenumber. Cosmo3DFlow achieves near-perfect alignment with ground truth across all scales.</em>
  <br><br>
  <img src="docs/figures/quantitative_metrics.png" alt="Physics validation metrics" width="800"/>
</p>

### Quantitative Results

<details>
<summary>All datasets × resolutions (Ours / Diffusion · bold = best)</summary>

#### Standard Latin Hypercube

| Resolution | VRMSE ↓ | Corr ↑ | PS R² ↑ | Transfer Fn ↑ |
|---|---|---|---|---|
| 128³ | **0.50** / 0.63 | **0.88** / 0.82 | **0.99** / 0.70 | **0.99** / 0.80 |
| 64³  | **0.47** / 0.68 | **0.92** / 0.89 | **0.98** / 0.59 | **0.98** / 0.59 |
| 32³  | **0.34** / 0.82 | **0.96** / 0.85 | **0.95** / 0.48 | **0.95** / 0.48 |

#### Big Sobol Sequence

| Resolution | VRMSE ↓ | Corr ↑ | PS R² ↑ | Transfer Fn ↑ |
|---|---|---|---|---|
| 128³ | **0.62** / 0.64 | **0.80** / 0.79 | **0.99** / 0.84 | **0.95** / 0.88 |
| 64³  | **0.53** / 0.65 | 0.88 / 0.88 | **0.98** / 0.83 | **0.94** / 0.81 |
| 32³  | **0.37** / 0.79 | **0.95** / 0.85 | **0.95** / 0.48 | **0.94** / 0.71 |

#### Non-Gaussian fNL LH

| Resolution | VRMSE ↓ | Corr ↑ | PS R² ↑ | Transfer Fn ↑ |
|---|---|---|---|---|
| 128³ | **0.56** / 0.59 | **0.86** / 0.83 | 1.00 / 1.00 | 0.98 / 0.98 |
| 64³  | **0.47** / 0.57 | **0.93** / 0.89 | 1.00 / 1.00 | 0.99 / 0.99 |
| 32³  | **0.31** / 0.67 | **0.97** / 0.87 | **1.00** / 0.98 | **0.99** / 0.98 |

</details>

---

## Installation

```bash
git clone https://github.com/khairul-me/Cosmo3DFlow.git
cd Cosmo3DFlow
pip install -r requirements.txt
```

---

## Citation

```bibtex
@article{islam2026cosmo3dflow,
  title   = {Cosmo3DFlow: Wavelet Flow Matching for Spatial-to-Spectral
             Compression in Reconstructing the Early Universe},
  author  = {Islam, Md Khairul and Xia, Zeyu and Goudjil, Ryan and
             Wang, Jialu and Farahi, Arya and Fox, Judy},
  journal = {arXiv preprint arXiv:2602.10172},
  year    = {2026}
}
```

---

## Acknowledgments

Supported by the **NSF-Simons AI Institute for Cosmic Origins** (CosmicAI, Grant 2421782).  
*University of Virginia · University of Texas at Austin*
