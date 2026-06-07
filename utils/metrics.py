import numpy as np
from tqdm import tqdm

import scipy.stats as stats
from typing import Dict, List, Tuple
import json
import Pk_library as PKL

import math
import torch

from torch import Tensor
from typing import Optional, Sequence, Tuple
from sklearn.metrics import r2_score

def pspec(x, boxsize=1000.0, kmax=1.0, threads=1, mas='None'):
    # Pylians expects float32 arrays
    delta = x.astype(np.float32)
    
    # Calculate auto-power spectrum
    # axis=0: assumed axis for RSD (irrelevant for 1D monopole of a box)
    # MAS='None': assumes x is already a density field (no MAS correction needed)
    Pk_obj = PKL.Pk(delta, BoxSize=boxsize, axis=0, MAS=mas, threads=threads, verbose=False)
    
    # Extract k and Power
    k = Pk_obj.k1D
    PS = Pk_obj.Pk1D
    
    # Always exclude k=0 mode
    mask = (k > 0)
    if kmax is not None:
        mask = mask & (k <= kmax)
    return PS[mask], k[mask]

def cross_pspec(x, y, boxsize=1000.0, kmax=1.0, threads=1, mas=['None', 'None']):
    # Pylians expects float32 arrays
    delta1 = x.astype(np.float32)
    delta2 = y.astype(np.float32)
    
    # Calculate auto and cross-power spectra in one go
    Pk_obj = PKL.XPk(
        [delta1, delta2], BoxSize=boxsize, axis=0,
        MAS=mas, threads=threads
    )
    
    k = Pk_obj.k1D
    PS_xx = Pk_obj.Pk1D[:, 0]
    PS_yy = Pk_obj.Pk1D[:, 1]
    PS_xy = Pk_obj.PkX1D[:, 0]
    
    # Calculate Cross-Correlation Coefficient
    # Avoid division by zero if necessary
    PS = PS_xy / np.sqrt(np.clip(PS_xx * PS_yy, 1e-12, None))

    # Always exclude k=0 mode
    mask = (k > 0)
    if kmax is not None:
        mask = mask & (k <= kmax)
    return PS[mask], k[mask]

def isotropic_binning(
    shape: Sequence[int],
    bins: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Computes an isotropic binning over the frequency domain.

    Arguments:
        shape: The domain shape :math:`(L_1, ..., L_N)`.
        bins: The number of bins :math:`B`.

    Returns:
        The bin edges, counts and indices, with shape :math:`(B + 1)`, :math:`(B + 1)`
        and :math:`(L_1 x ... x L_N)`, respectively.
    """

    k = []

    for s in shape:
        k_i = torch.fft.fftfreq(s)
        k.append(k_i)

    k2 = map(torch.square, k)
    k2_iso = sum(torch.meshgrid(*k2, indexing="ij"))
    k_iso = torch.sqrt(k2_iso)

    if bins is None:
        bins = math.floor(math.sqrt(k_iso.ndim) * min(k_iso.shape) / 2)

    edges = torch.linspace(0, k_iso.max(), bins + 1)

    indices = torch.bucketize(k_iso.flatten(), edges)
    counts = torch.bincount(indices, minlength=bins + 1)

    return edges, counts, indices


def isotropic_power_spectrum(x: Tensor, spatial: int = 2) -> Tuple[Tensor, Tensor]:
    r"""Computes the isotropic power spectrum of a field.

    Arguments:
        x: A field tensor, with shape :math:`(*, L_1, ..., L_N)`.
        spatial: The number of spatial dimensions :math:`N`.

    Returns:
        The binned power spectrum and the frequency bins (in cycles per pixel), with
        shape :math:`(*, B)` and :math:`(B)`, respectively.
    """

    x = torch.as_tensor(x)

    batch, shape = x.shape[:-spatial], x.shape[-spatial:]

    # Binning
    edges, counts, indices = isotropic_binning(shape)

    # Power spectrum
    s = torch.fft.fftn(x, dim=tuple(range(-spatial, 0)), norm="ortho")
    p = torch.square(torch.abs(s))
    p = torch.flatten(p, start_dim=-spatial)

    p_iso = torch.zeros((*batch, *edges.shape), dtype=x.dtype)
    p_iso = p_iso.scatter_add(dim=-1, index=indices.expand_as(p), src=p)
    p_iso = p_iso / torch.clip(counts, min=1)

    return p_iso[..., 1:], edges[1:]

class CosmologyMetrics:
    """
    Compute cosmology-specific metrics for dark matter density fields.
    
    USAGE:
    - For P(k) and T(k): Pass PHYSICAL density fields δ = ρ/ρ̄ - 1
    - For C(k): Any consistent representation (it's scale-invariant)
    - For VRMSE: Pass GLOBALLY normalized fields

    Parameters
    ----------
    boxsize : float
        Box size in Mpc/h
    kmax : float
        Maximum k value for binning
    threads : int
        Number of CPU threads to use (Pylians only)

    """
    def __init__(
        self, boxsize: float = 1000.0, kmax=1.0, threads: int = 1
    ):
        self.boxsize = boxsize
        self.kmax = kmax
        self.threads = threads
        
        try:
            from nbodykit.lab import ArrayMesh, FFTPower
            # Disable nbodykit if multi-threading is requested, as Pylians handles it natively

            # always running if possible for correlation
            self.use_nbodykit = True # if threads == 1 else False
            if self.use_nbodykit:
                print('Using nbodykit')
            else:
                print(f'Using Pylians with {threads} threads')
        except:
            self.use_nbodykit = False
            print(f'Nbodykit is not available. Using Pylians with {threads} threads')
            
    def _normalize_global(self, field: np.ndarray) -> np.ndarray:
        """
        Apply global normalization if enabled.
        
        IMPORTANT: This is NOT local normalization (per-field).
        Global mean/std are computed ONCE from entire training set.
        """
        if self.use_global_norm:
            return (field - self.global_mean) / self.global_std
        return field
        
    def power_spectrum(self, field: np.ndarray, mas: str = 'None') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 1D power spectrum P(k).

        CRITICAL: Uses PHYSICAL or GLOBALLY normalized fields.
        NO local normalization applied (this would force all P(k) to same amplitude).

        Parameters
        ----------
        field : np.ndarray
            Physical density field δ or globally normalized field
            (This field is typically the sample or the truth, and `mas` applies to it).

        mas : str
            Mass-assignment scheme ('CIC', 'PCS', 'None', etc.)

        Returns
        -------
        PS : np.ndarray
            Power spectrum values
        k : np.ndarray
            Wave numbers
        """
        field = field.squeeze()
        # NO NORMALIZATION - use physical field as-is
        
        # nbodykit seems to give unstable power spectrum sometimes
        # if self.use_nbodykit:
        #     from nbodykit.lab import ArrayMesh, FFTPower
        #     mesh = ArrayMesh(field, BoxSize=self.boxsize)
        #     result = FFTPower(mesh, mode='1d', kmax=self.kmax)
        #     PS = result.power

        #     ps, k = PS['power'].real, PS['k']
        #     if self.kmax is None:
        #         return ps[1:], k[1:]  # Skip k=0
        #     else:
        #         mask = (k > 0) & (k <= self.kmax)
        #         return ps[mask], k[mask]
        # else:
        delta = field.astype(np.float32)

        Pk_obj = PKL.Pk(delta, BoxSize=self.boxsize, axis=0, MAS=mas, threads=self.threads, verbose=False)
        
        # Extract k and Power
        k = Pk_obj.k1D
        PS = Pk_obj.Pk1D
        
        # Always exclude k=0 mode
        mask = (k > 0)
        if self.kmax is not None:
            mask = mask & (k <= self.kmax)
        return PS[mask], k[mask]
    
    def cross_correlation(
        self,
        field1: np.ndarray, field2: np.ndarray,
        mas: List[str] = ['None', 'None']
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cross-correlation C(k) = P_12(k) / sqrt(P_1(k) * P_2(k)).
        
        NOTE: Cross-correlation is a correlation coefficient, so it's invariant
        to scaling. Both physical fields and globally normalized fields work.
        We optionally apply global normalization for numerical stability.
        
        Parameters
        ----------
        field1, field2 : np.ndarray
            Physical density fields δ or globally normalized fields
            (Typically, `field1` is the predicted sample and `field2` is the ground truth).

        mas : List[str]
            List of mass-assignment schemes for [field1, field2].
        
        Returns
        -------
        C_k : np.ndarray
            Cross-correlation coefficient
        k : np.ndarray
            Wave numbers
        """
        field1 = field1.squeeze()
        field2 = field2.squeeze()
        
        # NO NORMALIZATION - C(k) is scale-invariant
        
        if self.use_nbodykit:
            from nbodykit.lab import ArrayMesh, FFTPower
            mesh1 = ArrayMesh(field1, BoxSize=self.boxsize)
            mesh2 = ArrayMesh(field2, BoxSize=self.boxsize)
            
            result_12 = FFTPower(first=mesh1, mode='1d', second=mesh2, kmax=self.kmax)
            result_11 = FFTPower(first=mesh1, mode='1d', kmax=self.kmax)
            result_22 = FFTPower(first=mesh2, mode='1d', kmax=self.kmax)
            
            PS_12 = result_12.power['power'].real
            PS_11 = result_11.power['power'].real
            PS_22 = result_22.power['power'].real
            k = result_12.power['k']
            
            # Avoid division by zero
            # Add epsilon inside sqrt for stability, matching Pylians' implementation
            denominator = np.sqrt(np.clip(PS_11 * PS_22, 1e-12, None))
            
            C_k = PS_12 / denominator

            # nbodykit's k already excludes k=0. Just apply kmax if specified.
            mask = (k > 0) # Redundant for nbodykit, but explicit
            if self.kmax is not None:
                mask = mask & (k <= self.kmax)
            return C_k[mask], k[mask]
        else:
            delta1 = field1.astype(np.float32)
            delta2 = field2.astype(np.float32)
            
            # Calculate auto and cross-power spectra in one go
            Pk_obj = PKL.XPk(
                [delta1, delta2], BoxSize=self.boxsize, axis=0,
                MAS=mas, threads=self.threads
            )
            
            # Extract k
            k = Pk_obj.k1D
            
            # Extract Power Spectra
            PS_xx = Pk_obj.Pk1D[:, 0]
            PS_yy = Pk_obj.Pk1D[:, 1]
            PS_xy = Pk_obj.PkX1D[:, 0]
            
            # Calculate Cross-Correlation Coefficient
            # Avoid division by zero if necessary
            C_k = PS_xy / np.sqrt(np.clip(PS_xx * PS_yy, 1e-12, None))

            # Always exclude k=0 mode
            mask = (k > 0)
            if self.kmax is not None:
                mask = mask & (k <= self.kmax)
            return C_k[mask], k[mask]
    
    def transfer_function(
        self,
        field: np.ndarray, truth: np.ndarray,
        sample_mas: str = 'None',
        truth_mas: str = 'None'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute transfer function T(k) = sqrt(P_sample(k) / P_truth(k)).
        
        CRITICAL: Must use same normalization as power_spectrum().
        If you use physical fields here, P(k) will be physical.
        T(k) measures whether model preserves mode amplitudes correctly.
        
        With local normalization, T(k) ≈ 1.0 always (meaningless).
        
        Parameters
        ----------
        field : np.ndarray
            Predicted field (physical δ or globally normalized)
        truth : np.ndarray  
            Ground truth field (same normalization as field)
            (Here, `field` is explicitly the sample and `truth` is the ground truth).

        sample_mas : str
            Mass-assignment scheme for the predicted field.
        truth_mas : str
            Mass-assignment scheme for the ground truth field.
        
        Returns
        -------
        T_k : np.ndarray
            Transfer function (should be ~1.0 for good reconstruction)
        k : np.ndarray
            Wave numbers
        """
        P_sample, k_sample = self.power_spectrum(field, mas=sample_mas)
        P_truth, k_truth = self.power_spectrum(truth, mas=truth_mas)
        
        # Avoid division by zero
        P_truth_safe = np.clip(P_truth, 1e-12, None)
        T_k = np.sqrt(P_sample / P_truth_safe)
        return T_k, k_sample
    
    def power_spectrum_rmse_single(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        spatial: int = 3,
        n_bands: int = 3,
        eps: float = 1e-12,
    ) -> Dict[str, float]:
        """
        Power Spectrum RMSE as used in Ohana et al. (2023).
        
        This metric is OK with global normalization because it compares
        ratios of power spectra, not absolute values.
        
        Parameters
        ----------
        u : torch.Tensor
            Ground truth field
        v : torch.Tensor
            Generated field
        spatial : int
            Number of spatial dimensions
        n_bands : int
            Number of frequency bands
        eps : float
            Numerical stability parameter
        
        Returns
        -------
        dict
            RMSE scores for low/mid/high frequency bands
        """

        # Isotropic power spectra
        Pu, k = isotropic_power_spectrum(u, spatial=spatial)
        Pv, _ = isotropic_power_spectrum(v, spatial=spatial)

        # Relative power
        ratio = Pv / (Pu + eps)

        # Log-spaced frequency bands
        log_k = torch.log(k)
        band_edges = np.linspace(
            log_k.min(), log_k.max(), n_bands + 1
        )

        rmse = {}

        for i in range(n_bands):
            mask = (log_k >= band_edges[i]) & (log_k < band_edges[i + 1])
            if mask.any():
                r = ratio[..., mask]
                rmse_val = torch.sqrt(torch.mean((r - 1.0) ** 2))
            else:
                rmse_val = torch.tensor(float("nan"))

            rmse[["low", "mid", "high"][i]] = rmse_val.item()

        return rmse

class ValidationMetrics:
    """
    Validation metrics for cosmological density field reconstruction.
    
    USAGE:
    ------
    metrics = ValidationMetrics(boxsize=1000.0)
    
    # For P(k), T(k): Use physical density fields δ = ρ/ρ̄ - 1
    samples_delta = load_physical_samples()
    truth_delta = load_physical_truth()
    
    # For VRMSE: Use globally normalized fields
    samples_global = (samples - global_mean) / global_std
    truth_global = (truth - global_mean) / global_std
    
    results = metrics.compute_all_metrics(samples_delta, truth_delta, 
                                         samples_global, truth_global)
    """
    
    def __init__(
        self, 
        boxsize: float = 1000.0, 
        kmax: float = 0.4,
        threads: int = 1,
        truth_mas: str = 'None',
        compute_diversity_metrics: bool = False
    ):
        self.cosmo_metrics = CosmologyMetrics(
            boxsize=boxsize,
            kmax=kmax,
            threads=threads
        )
        self.kmax = kmax
        self.truth_mas = truth_mas
        self.compute_diversity_metrics = compute_diversity_metrics
        
    def cross_correlation_score(
        self, 
        samples: np.ndarray, 
        truth: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute cross-correlation C(k) between samples and truth.
        Higher C(k) means better reconstruction.
        
        Parameters
        ----------
        samples : np.ndarray (n_samples, D, H, W) - Predicted fields.
        truth : np.ndarray (D, H, W) - Ground truth field.
        Expected range: [0, 1], with 1 being perfect correlation.
        """
        cross_corrs = []
        for sample in samples:
            C_k, k = self.cosmo_metrics.cross_correlation(
                sample, truth, mas=['None', self.truth_mas]
            )
            cross_corrs.append(np.nanmean(C_k))

        return {
            'mean': float(np.nanmean(cross_corrs)),
            'std': float(np.nanstd(cross_corrs)),
            'score': float(np.nanmean(cross_corrs))
        }
    
    def transfer_function_accuracy(
        self, 
        samples: np.ndarray, 
        truth: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute transfer function T(k). Should be ~1.0 for perfect reconstruction.
        
        Parameters
        ----------
        samples : np.ndarray (n_samples, D, H, W) - Predicted fields.
        truth : np.ndarray (D, H, W) - Ground truth field.
        CRITICAL: This metric is ONLY meaningful with physical or globally
        normalized fields. With local normalization, T(k) ≈ 1.0 always.
        
        Expected behavior:
        - Good model: mean(|T(k) - 1.0|) < 0.1
        - Bad model: mean(|T(k) - 1.0|) >> 0.1
        """
        transfer_deviations = []
        for sample in samples:
            T_k, k = self.cosmo_metrics.transfer_function(
                sample, truth, sample_mas='None', truth_mas=self.truth_mas
            )
            deviation = np.nanmean(np.abs(T_k - 1.0))
            transfer_deviations.append(deviation)
        
        return {
            'mean': float(1.0 - np.clip(np.nanmean(transfer_deviations), 0, 1)),
            'std': float(np.nanstd(transfer_deviations)),
            'score': float(1.0 - np.clip(np.nanmean(transfer_deviations), 0, 1))
        }

    def power_spectrum_r2(self, samples, truth):
        scores = []
        # Recompute P(k) for truth with correct MAS
        # Parameters
        # ----------
        # samples : np.ndarray (n_samples, D, H, W) - Predicted fields.
        # truth : np.ndarray (D, H, W) - Ground truth field.
        ps_truth, _ = self.cosmo_metrics.power_spectrum(truth, mas=self.truth_mas)
        for sample in samples:
            ps_sample, _ = self.cosmo_metrics.power_spectrum(sample, mas='None')
            r2 = r2_score(ps_truth, ps_sample)
            scores.append(r2)

        return {
            'mean': float(np.nanmean(scores)),
            'std': float(np.nanstd(scores)),
            'score': float(np.nanmean(scores))
        }

    def sample_variance_metrics(self, samples: np.ndarray, truth: np.ndarray) -> Dict[str, Dict]:
        """
        Computes metrics related to the variance across multiple generated samples.
        This is a measure of the model's posterior diversity.

        Returns a dictionary with three metrics:
        - 'absolute': The mean pixel-wise variance in physical units. A higher value
                      indicates greater absolute diversity. This can be very small.
        - 'relative_to_mean': The mean pixel-wise variance normalized by the square of the
                              sample mean. This is a scale-independent measure but can be
                              unstable in regions where the mean is close to zero.
        - 'relative_to_truth_variance': The mean sample variance normalized by the variance
                                        of the ground truth field. A value around 1.0 is
                                        desirable, indicating that the sample diversity
                                        is on par with the spatial diversity of the truth.
        """
        if samples.shape[0] <= 1:
            # Variance is undefined for a single sample.
            return {
                'absolute': {'mean': 0.0, 'std': 0.0, 'score': 0.0},
                'relative_to_mean': {'mean': 0.0, 'std': 0.0, 'score': 0.0},
                'relative_to_truth_variance': {'mean': 0.0, 'std': 0.0, 'score': 0.0}
            }

        # 1. Absolute sample variance
        variance_map = np.var(samples, axis=0)
        mean_absolute_variance = np.mean(variance_map)

        # 2. Relative to mean (old 'relative')
        sample_mean = np.mean(samples, axis=0)
        valid_pixels_mask = np.abs(sample_mean) > 1e-6
        mean_relative_to_mean_variance = 0.0
        if np.any(valid_pixels_mask):
            relative_variance_map = variance_map[valid_pixels_mask] / (sample_mean[valid_pixels_mask]**2)
            mean_relative_to_mean_variance = np.mean(relative_variance_map)

        # 3. Relative to truth variance (new metric)
        truth_variance = np.var(truth)
        mean_relative_to_truth_variance = 0.0
        if truth_variance > 1e-9:
            mean_relative_to_truth_variance = mean_absolute_variance / truth_variance

        return {
            'absolute': {'mean': float(mean_absolute_variance), 'std': 0.0, 'score': float(mean_absolute_variance)},
            'relative_to_mean': {'mean': float(mean_relative_to_mean_variance), 'std': 0.0, 'score': float(mean_relative_to_mean_variance)},
            'relative_to_truth_variance': {'mean': float(mean_relative_to_truth_variance), 'std': 0.0, 'score': float(mean_relative_to_truth_variance)}
        }

    def variance_power_spectrum(self, samples: np.ndarray, truth: np.ndarray) -> Dict[str, Dict]:
        """
        Computes the variance of the model's posterior in Fourier space, normalized
        by the power spectrum of the truth field. This is the "Diversity Power Spectrum".

        This provides a scale-dependent, dimensionless measure of relative sample diversity,
        answering: "At which scales (k) is the model's diversity lacking compared to the truth?"

        It returns a binned ratio:
        Ratio(k) = ( <P(sample_i)> - P(<sample_i>) ) / P(truth)
        where <.> is the average over samples.

        - A value << 1.0 in a k-band indicates under-confidence at those scales.
        - A value ~ 1.0 is desirable, suggesting the diversity power matches the
          signal power of the truth field at those scales.
        - The mean of this metric across scales should be comparable to the
          `sample_variance_relative_to_truth_variance` metric.
        """
        if samples.shape[0] <= 1:
            return {}

        # 1. Compute P(k) for each sample
        sample_pspecs = []
        k = None
        for sample in samples:
            ps, k_current = self.cosmo_metrics.power_spectrum(sample, mas='None')
            if k_current.size == 0: continue
            if k is None: k = k_current
            assert np.allclose(k, k_current), "k-bins from power spectra do not match between samples."
            sample_pspecs.append(ps)
        
        if not sample_pspecs or k is None: return {}
        sample_pspecs = np.array(sample_pspecs)
        
        # 2. Average P(k) of samples
        mean_of_ps = np.mean(sample_pspecs, axis=0)

        # 3. Compute P(k) of the mean of samples
        mean_of_samples = np.mean(samples, axis=0)
        ps_of_mean, k_mean = self.cosmo_metrics.power_spectrum(mean_of_samples, mas='None')
        if k_mean.size == 0: return {}

        assert np.allclose(k, k_mean), "k-bins from power spectra do not match."

        # 4. Diversity Power Spectrum: P_div(k) = <P(s_i)> - P(<s_i>)
        ps_diversity = mean_of_ps - ps_of_mean

        # 5. Power spectrum of the truth field for normalization
        ps_truth, k_truth = self.cosmo_metrics.power_spectrum(truth, mas=self.truth_mas)
        if k_truth.size == 0: return {}
        
        assert np.allclose(k, k_truth), "k-bins from power spectra do not match."

        # 6. Compute the dimensionless ratio of diversity power to truth power
        relative_ps = ps_diversity / (ps_truth + 1e-12)

        # Bin the power into frequency bands
        log_k = np.log(k)
        n_bands = 3
        try:
            band_edges = np.linspace(log_k.min(), log_k.max(), n_bands + 1)
        except ValueError:
            return {}

        binned_power = {}
        bands = ['low_k', 'mid_k', 'high_k']
        for i in range(n_bands):
            mask = (log_k >= band_edges[i]) & (log_k < band_edges[i+1])
            if mask.any():
                mean_power_in_band = np.mean(relative_ps[mask])
            else:
                mean_power_in_band = 0.0
            
            binned_power[bands[i]] = {
                'mean': float(mean_power_in_band),
                'std': 0.0, # Only one value per simulation
                'score': float(mean_power_in_band)
            }
        return binned_power

    def calibration_error(self, samples: np.ndarray, truth: np.ndarray, epsilon=1e-9) -> Dict[str, float]:
        """
        Computes a calibration error metric based on the log-ratio of sample variance 
        to the mean squared error of the sample mean. A score close to 0 is better, 
        indicating the sample spread matches the error of the sample mean.

        For a well-calibrated probabilistic model, the variance of the posterior predictive
        distribution should match the expected squared error of the mean prediction.

        Reference:
        Kuleshov, V., Fenner, N., & Ermon, S. (2018). "Accurate Uncertainties for
        Deep Learning Using Calibrated Regression." ICML.
        """
        if samples.shape[0] <= 1:
            return {'mean': float('nan'), 'std': 0.0, 'score': float('nan')}

        sample_mean = np.mean(samples, axis=0)
        sample_variance = np.var(samples, axis=0)

        mean_sample_variance = np.mean(sample_variance)
        mean_squared_error = np.mean((sample_mean - truth)**2)

        log_calib_error = np.abs(np.log(mean_sample_variance + epsilon) - np.log(mean_squared_error + epsilon))

        return {
            'mean': float(log_calib_error),
            'std': 0.0,
            'score': float(log_calib_error) # Lower is better
        }

    def crps_score(self, samples: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
        """
        Computes the ensemble Continuous Ranked Probability Score (CRPS).
        This is a proper scoring rule that generalizes the Mean Absolute Error to 
        probabilistic forecasts, assessing both calibration and sharpness. A lower 
        score is better.

        The score is calculated as: CRPS = MAE - 0.5 * MAD, where MAE is the mean
        absolute error between the samples and the truth, and MAD is the mean
        absolute difference between all pairs of samples.

        Reference:
        Gneiting, T., & Raftery, A. E. (2007). "Strictly Proper Scoring Rules,
        Prediction, and Estimation." Journal of the American Statistical Association.
        """
        if samples.shape[0] <= 1:
            return {'mean': float('nan'), 'std': 0.0, 'score': float('nan')}

        n_samples = samples.shape[0]
        
        # Term 1: Mean absolute error between samples and truth
        mae = np.mean(np.abs(samples - truth))

        # Term 2: Mean absolute difference between all pairs of samples
        sorted_samples = np.sort(samples, axis=0)
        i_vals = np.arange(1, n_samples + 1).reshape(n_samples, 1, 1, 1)
        sum_term = np.sum((2 * i_vals - n_samples - 1) * sorted_samples, axis=0)
        mean_abs_diff = np.mean((2 / (n_samples * n_samples)) * sum_term)
        
        crps = mae - 0.5 * mean_abs_diff
        
        return {'mean': float(crps), 'std': 0.0, 'score': float(crps)}
        
    def prediction_interval_coverage(self, samples: np.ndarray, truth: np.ndarray, confidence_level: float = 0.9) -> Dict[str, float]:
        """
        Computes the Prediction Interval Coverage Probability (PICP).
        This metric directly measures the percentage of spatial locations where the
        ground truth falls within the prediction interval formed by the generated samples.
        A value close to the `confidence_level` indicates good coverage.

        For example, a `confidence_level` of 0.9 (90%) means we expect the true value
        to be within the 5th and 95th percentile of the samples 90% of the time.

        Reference:
        Gneiting, T., & Raftery, A. E. (2007). "Strictly Proper Scoring Rules,
        Prediction, and Estimation." Journal of the American Statistical Association.
        (While not directly defining PICP, it's a fundamental concept in probabilistic
        forecasting, often discussed in conjunction with proper scoring rules like CRPS).
        """
        if samples.shape[0] <= 1:
            return {'mean': float('nan'), 'std': 0.0, 'score': float('nan')}

        alpha = 1.0 - confidence_level
        lower_percentile = alpha / 2.0 * 100
        upper_percentile = (1.0 - alpha / 2.0) * 100

        lower_bound = np.percentile(samples, lower_percentile, axis=0)
        upper_bound = np.percentile(samples, upper_percentile, axis=0)

        coverage = ((truth >= lower_bound) & (truth <= upper_bound)).mean()

        return {'mean': float(coverage), 'std': 0.0, 'score': float(coverage)} # Higher is better, ideally close to confidence_level

    def power_spectrum_rmse(
        self, samples: np.ndarray, truth: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Compute power spectrum RMSE in different frequency bands.
        Uses logarithmic bands as in Ohana et al. (2023).
        """
        
        bands = ['low', 'mid', 'high']
        results = {band: [] for band in bands}
      
        for sample in samples:
            score = self.cosmo_metrics.power_spectrum_rmse_single(
                torch.from_numpy(truth), 
                torch.from_numpy(sample)
            )
            for band in bands:
                results[band].append(score[band])
        
        new_results = {}
        for band in bands:
            new_results[band] = {
                'mean': np.mean(results[band]),
                'std': np.std(results[band]),
                'score': np.mean(results[band])
            }
        
        return new_results
        
    def vrmse_score(
        self, samples: np.ndarray, truth: np.ndarray, epsilon=1e-6
    ) -> Dict[str, float]:
        """
        Computes Variance-Normalized RMSE (VRMSE).
        
        Formula: VRMSE = sqrt(MSE(u, v) / (Var(u) + epsilon))
        
        CRITICAL: Requires GLOBALLY normalized fields where Var(truth) ≈ 1.
        
        Parameters
        ----------
        samples : np.ndarray
            Predicted fields (globally normalized), shape (n_samples, D, H, W)
        truth : np.ndarray
            Ground truth field (globally normalized), shape (D, H, W)
        epsilon : float
            Numerical stability (default 1e-6)
        
        Returns
        -------
        dict
            VRMSE statistics across samples
        """
        
        truth_mean = np.mean(truth)
        truth_var = np.mean((truth - truth_mean) ** 2)
        denominator = truth_var + epsilon
        
        scores = []
        for v in samples:
            mse = np.mean((truth - v) ** 2)
            vrmse = np.sqrt(mse / denominator)
            scores.append(vrmse)
        
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'score': float(np.mean(scores))
        }
        
    def compute_all_metrics(
        self, 
        samples_physical: np.ndarray, 
        truth_physical: np.ndarray,
        samples_global_norm: Optional[np.ndarray] = None,
        truth_global_norm: Optional[np.ndarray] = None
    ) -> Dict[str, Dict]:
        """
        Compute all validation metrics for a single example.
        
        Parameters
        ----------
        samples_physical : np.ndarray
            Predicted samples in physical units δ, shape (n_samples, D, H, W)
            Used for: P(k), T(k), C(k)
        truth_physical : np.ndarray
            Ground truth in physical units δ, shape (D, H, W)
            Used for: P(k), T(k), C(k)
        samples_global_norm : np.ndarray, optional
            Globally normalized samples for VRMSE
            If None, will use physical samples (may give different scale)
        truth_global_norm : np.ndarray, optional
            Globally normalized truth for VRMSE
            If None, will use physical truth
        
        Returns
        -------
        dict
            All metric results
        """
        # Use globally normalized fields for VRMSE if provided
        if samples_global_norm is None:
            samples_global_norm = samples_physical
        if truth_global_norm is None:
            truth_global_norm = truth_physical
        
        metrics = {
            'cross_correlation': self.cross_correlation_score(samples_physical, truth_physical),
            'transfer_function': self.transfer_function_accuracy(samples_physical, truth_physical),
            'vrmse': self.vrmse_score(samples_global_norm, truth_global_norm),
            'power_spectrum_rmse': self.power_spectrum_rmse(samples_physical, truth_physical),
            'r2_score': self.power_spectrum_r2(samples_global_norm, truth_global_norm)
        }
        
        if self.compute_diversity_metrics:
            if samples_physical.shape[0] > 1:
                metrics['sample_variance'] = self.sample_variance_metrics(samples_physical, truth_physical)
                metrics['variance_power_spectrum'] = self.variance_power_spectrum(samples_physical, truth_physical)
                metrics['calibration_error'] = self.calibration_error(samples_physical, truth_physical)
                metrics['crps_score'] = self.crps_score(samples_physical, truth_physical)
                metrics['prediction_interval_coverage'] = self.prediction_interval_coverage(samples_physical, truth_physical)
            else:
                print("Warning: Cannot compute diversity metrics with only one sample per example.")
        
        return metrics


class ValidationSuite:
    """Run validation across entire validation dataset."""
    
    def __init__(
        self, boxsize: float = 1000.0,
        kmax: float = 0.4,
        threads: int = 1,
        truth_mas: str = 'None',
        compute_diversity_metrics: bool = False
    ):
        self.metrics_computer = ValidationMetrics(
            boxsize=boxsize,
            kmax=kmax,
            threads=threads,
            truth_mas=truth_mas,
            compute_diversity_metrics=compute_diversity_metrics
        )
        self._reset_accumulators()

    def _reset_accumulators(self):
        """Reset rolling statistics accumulators."""
        self.n = 0
        self.running_stats = {}
        self.per_example_results = []

    def _update_rolling_stats(self, metric_value: float, metric_name: str):
        """
        Update rolling statistics using Welford's online algorithm.
        
        This computes mean and variance incrementally without storing all values.
        
        Parameters
        ----------
        metric_value : float
            New metric value to add
        metric_name : str
            Name of the metric
        """
        if metric_name not in self.running_stats:
            self.running_stats[metric_name] = {
                'n': 0,
                'mean': 0.0,
                'M2': 0.0,  # Sum of squared differences from mean
                'values': []  # Store only for median calculation
            }
        
        stats = self.running_stats[metric_name]
        stats['n'] += 1
        
        # Welford's algorithm for online mean and variance
        delta = metric_value - stats['mean']
        stats['mean'] += delta / stats['n']
        delta2 = metric_value - stats['mean']
        stats['M2'] += delta * delta2
        
        # Store value for median (we need to keep these, but they're just scalars)
        stats['values'].append(metric_value)

    def add_example(
        self,
        samples_physical: np.ndarray,
        truth_physical: np.ndarray,
        samples_global_norm: Optional[np.ndarray] = None,
        truth_global_norm: Optional[np.ndarray] = None,
        save_per_example: bool = False
    ):
        """
        Add a single validation example and update rolling statistics.
        
        This method is memory-efficient as it processes one example at a time
        and only keeps running statistics, not all samples.
        
        Parameters
        ----------
        samples_physical : np.ndarray
            Physical density fields for this example
        truth_physical : np.ndarray
            Ground truth physical field
        samples_global_norm : np.ndarray, optional
            Globally normalized samples for VRMSE
        truth_global_norm : np.ndarray, optional
            Globally normalized truth for VRMSE
        save_per_example : bool
            If True, save individual example results (uses more memory)
        """
        # Compute metrics for this example
        example_metrics = self.metrics_computer.compute_all_metrics(
            samples_physical, truth_physical,
            samples_global_norm, truth_global_norm
        )
        
        # Update rolling statistics for each metric
        for metric_name, metric_dict in example_metrics.items():
            if 'score' in metric_dict:
                # Simple metric with single score
                full_name = metric_name
                self._update_rolling_stats(metric_dict['score'], full_name)
            else:
                # Nested metric (like power_spectrum_rmse)
                for sub_metric_name, sub_dict in metric_dict.items():
                    full_name = f"{metric_name}_{sub_metric_name}"
                    self._update_rolling_stats(sub_dict['score'], full_name)
        
        # Optionally save individual results
        if save_per_example:
            self.per_example_results.append(example_metrics)
        
        self.n += 1

    def add_example_metrics(self, example_metrics: Dict, sample_no: int):
        """
        Add pre-computed metrics for a single example and update rolling statistics.
        
        Parameters
        ----------
        example_metrics : dict
            Pre-computed metrics for one example.
        sample_no : int
            The simulation ID for this example.
        """
        # Update rolling statistics for each metric
        for metric_name, metric_dict in example_metrics.items():
            if 'score' in metric_dict:
                # Simple metric with single score
                full_name = metric_name
                self._update_rolling_stats(metric_dict['score'], full_name)
            else:
                # Nested metric (like power_spectrum_rmse)
                for sub_metric_name, sub_dict in metric_dict.items():
                    full_name = f"{metric_name}_{sub_metric_name}"
                    self._update_rolling_stats(sub_dict['score'], full_name)
        
        # Store individual results with sample_no
        self.per_example_results.append({'sample_no': sample_no, 'metrics': example_metrics})
        
        self.n += 1

    def _get_current_stats(self) -> Dict[str, float]:
        """Get current statistics (for progress bar updates)."""
        stats = {}
        for metric_name in ['cross_correlation', 'transfer_function', 'vrmse', 'r2_score']:
            if metric_name in self.running_stats:
                stats[metric_name] = self.running_stats[metric_name]['mean']
        return stats
    
    def _finalize_stats(self) -> Dict:
        """Finalize and return aggregate statistics."""
        aggregate_metrics = {}
        
        for metric_name, stats in self.running_stats.items():
            n = stats['n']
            mean = stats['mean']
            
            # Compute standard deviation from M2
            if n > 1:
                variance = stats['M2'] / (n - 1)
                std = np.sqrt(variance)
            else:
                std = 0.0
            
            # Compute median from stored values
            median = float(np.median(stats['values']))
            
            aggregate_metrics[metric_name] = {
                'mean': float(mean),
                'std': float(std),
                'median': median
            }
        
        final_results = {'summary': aggregate_metrics}
        if self.per_example_results:
            # Sort by sample_no for consistent output
            self.per_example_results.sort(key=lambda x: x['sample_no'])
            final_results['per_example'] = self.per_example_results
        return final_results
    
    def save_results(self, results: Dict, output_path: str):
        """Save validation results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        # print(f"Results saved to {output_path}")
    
    def print_summary(self, results: Dict):
        """Print formatted summary of validation results."""
        summary = results.get('summary', results)

        print("\n" + "="*60)
        print("VALIDATION RESULTS SUMMARY")
        print("="*60)
        
        print("\nDETAILED METRICS:")
        print("-" * 60)

        for metric_name, values in summary.items():
            print(f"\n{metric_name.upper().replace('_', ' ')}:")
            print(f"  Mean:   {values['mean']:.4f} ± {values['std']:.4f}")
            print(f"  Median: {values['median']:.4f}")
        
        print("\n" + "="*60)