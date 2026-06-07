from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import os
import torch, random
from tqdm import tqdm

def list_files(directory, ext='h5'):
    return sorted([str(file) for file in Path(directory).rglob(f"*.{ext}")])

def get_filepath(sample_no, file_type):
    if file_type == 'standard_z0_128':
        # return f"Train_z0_2000/{sample_no}_z0.npy"
        return f"Quijote_processed/Z0_128/{sample_no}_z0.npy"
    elif file_type == 'standard_ic_128':
        # return f"Train_z127_from_IC_2000/df_m_z=127_sim{sample_no}.npy"
        return f"Quijote_processed/IC_128/df_m_z=127_sim{sample_no}.npy"
    elif file_type == 'standard_z0_256':
        return f"Quijote_processed/Z0_256/{sample_no}.npy"
    elif file_type == 'standard_ic_256':
        return f"Quijote_processed/IC_256/{sample_no}.npy"
    elif file_type == 'standard_z0_512':
        return f"Quijote_processed/Z0_512/{sample_no}.npy"
    elif file_type == 'standard_ic_512':
        return f"Quijote_processed/IC_512/{sample_no}.npy"
    elif file_type == 'standard_z0_32':
        return f'Quijote_processed/Z0_32/{sample_no}.npy'
    elif file_type == 'standard_z0_64':
        return f'Quijote_processed/Z0_64/{sample_no}.npy'
    elif file_type == 'standard_ic_32':
        return f'Quijote_processed/IC_32/{sample_no}.npy'
    elif file_type == 'standard_ic_64':
        return f'Quijote_processed/IC_64/{sample_no}.npy'
    elif file_type == 'lc_ic_32':
        return f'latin_hypercube_LC_processed/IC_32/{sample_no}.npy'
    elif file_type == 'lc_ic_64':
        return f'latin_hypercube_LC_processed/IC_64/{sample_no}.npy'
    elif file_type == 'lc_ic_128':
        return f'latin_hypercube_LC_processed/IC_128/{sample_no}.npy'
    elif file_type == 'lc_z0_32':
        return f'latin_hypercube_LC_processed/Z0_32/{sample_no}.npy'
    elif file_type == 'lc_z0_64':
        return f'latin_hypercube_LC_processed/Z0_64/{sample_no}.npy'
    elif file_type == 'lc_z0_128':
        return f'latin_hypercube_LC_processed/Z0_128/{sample_no}.npy'
    elif file_type == 'bsq_ic_32':
        return f'BSQ_Processed/IC_32/{sample_no}.npy'
    elif file_type == 'bsq_ic_64':
        return f'BSQ_Processed/IC_64/{sample_no}.npy'
    elif file_type == 'bsq_ic_128':
        return f'BSQ_Processed/IC_128/{sample_no}.npy'
    elif file_type == 'bsq_z0_32':
        return f'BSQ_Processed/Z0_32/{sample_no}.npy'
    elif file_type == 'bsq_z0_64':
        return f'BSQ_Processed/Z0_64/{sample_no}.npy'
    elif file_type == 'bsq_z0_128':
        return f'BSQ_Processed/Z0_128/{sample_no}.npy'
    elif file_type == 'eq_ic_32':
        return f'latin_hypercube_EQ_processed/IC_32/{sample_no}.npy'
    elif file_type == 'eq_ic_64':
        return f'latin_hypercube_EQ_processed/IC_64/{sample_no}.npy'
    elif file_type == 'eq_ic_128':
        return f'latin_hypercube_EQ_processed/IC_128/{sample_no}.npy'
    elif file_type == 'eq_z0_32':
        return f'latin_hypercube_EQ_processed/Z0_32/{sample_no}.npy'
    elif file_type == 'eq_z0_64':
        return f'latin_hypercube_EQ_processed/Z0_64/{sample_no}.npy'
    elif file_type == 'eq_z0_128':
        return f'latin_hypercube_EQ_processed/Z0_128/{sample_no}.npy'
    elif file_type == 'hr_z0_1024':
        return f'Quijote_HR_processed/Z0_1024/{sample_no}.npy'
    elif file_type == 'hr_ic_1024':
        return f'Quijote_HR_processed/IC_1024/{sample_no}.npy'
    elif file_type == 'hr_z0_256':
        return f'Quijote_HR_processed/Z0_256/{sample_no}.npy'
    elif file_type == 'hr_ic_256':
        return f'Quijote_HR_processed/IC_256/{sample_no}.npy'
    elif file_type == 'hr_z0_512':
        return f'Quijote_HR_processed/Z0_512/{sample_no}.npy'
    elif file_type == 'hr_ic_512':
        return f'Quijote_HR_processed/IC_512/{sample_no}.npy'
    elif file_type == 'halo_128':
        return f'Quijote_processed/halo_LH_128/halo_lh_{sample_no:04d}.npy'
    else:
        raise ValueError(f"Unknown file type: {file_type}")
    
# {input_type: [mean, std]}
stats_dict = {
    'standard_z0_128': [-0.1235, 0.3096], # this is after np.log10(data + 1) transformation
    'standard_ic_128': [0, 0.00927],
    'standard_z0_256': [-0.1714, 0.3548],
    'standard_ic_256': [0, 0.0142],
    'standard_z0_512': [-0.2981, 0.4608],
    'standard_ic_512': [0, 0.02152],
    'halo_128': [-1.0941, 1.7070], # after applying log(data+1+1e-5) # [0.2157, 0.2432], # after applying np.log10(data + 2), +2 because data mostly centers around -1
    'recon': [-0.1724, 0.356],
    # 'camels_z0': [-0.826724, 0.718647],
    # 'camels_z127': [0, 0.04295]
    # 'camels_z0': [-0.655149 , 0.575375],
    # 'camels_z127': [0, 0.034408]
    'camels_z0': [-0.518061, 0.540111],
    'camels_z127': [0, 0.026746],
    'standard_z0_32': [-0.0106, 0.0955],
    'standard_z0_64': [-0.0335, 0.1681],
    'standard_ic_32': [0, 0.0033],
    'standard_ic_64': [0, 0.0057],
    'hr_z0_256': [-0.1760, 0.3604],
    'hr_ic_256': [0, 0.0139],
    'hr_z0_512': [-0.3051, 0.4463],
    'hr_ic_512': [0, 0.02],
    'hr_z0_1024': [-0.3985, 0.5166],
    'hr_ic_1024': [-0.0000, 0.0243],

    # lc dataset
    'lc_z0_32': [-0.0093, 0.0895],
    'lc_z0_64': [-0.0334, 0.1683],
    'lc_z0_128': [-0.0891, 0.2674],
    'lc_ic_32': [0, 0.0029],
    'lc_ic_64': [0, 0.0055],
    'lc_ic_128': [0, 0.0091],

    # bsq dataset
    'bsq_z0_32': [-0.0108, 0.0963],
    'bsq_z0_64': [-0.0338, 0.1688],
    'bsq_z0_128': [-0.0854, 0.2621],
    'bsq_ic_32': [0, 0.0033],
    'bsq_ic_64': [0, 0.0057],
    'bsq_ic_128': [0, 0.0092],

    # eq dataset
    'eq_z0_32': [0.9287, 0.1465],
    'eq_z0_64': [0.8908, 0.2239],
    'eq_z0_128': [0.8352, 0.2910],
    'eq_ic_32': [0, 0.0033],
    'eq_ic_64': [0, 0.0057],
    'eq_ic_128': [0, 0.0093]
}

raw_stats = {
    'standard_z0_32': [0, 0.2279],
    'standard_z0_64': [0, 0.4279],
    'standard_z0_128': [0, 1.0393]
}

arcsinh_stats = {
    'standard_z0_32': [-0.0283, 0.7559],
    'standard_z0_64': [-0.0475, 0.7342],
    'standard_z0_128': [-0.0686, 0.6143]
}

class SimulationDataset(Dataset):
    def __init__(
        self, root, idx_list,
        input_type='standard_z0_128', target_type='standard_ic_128',
        normalize=True, augment=True,
        log_transform_input=True,
        arcsinh_transform_input=False,
        return_cosmo=False,
        cosmo_params_path=None,
        local_normalize_target=False, # specially helpful for diffusion models
        min_max_target= False
    ):
        self.root = root
        self.normalize = normalize
        self.input_type = input_type
        self.target_type = target_type
        
        self.log_transform_input = log_transform_input # mainly for the z0 data
        self.idx_list = list(idx_list)
        self.return_cosmo = return_cosmo
        self.local_normalize_target = local_normalize_target
        self.arcsinh_transform_input = arcsinh_transform_input
        self.min_max_target = min_max_target

        assert not (log_transform_input and arcsinh_transform_input), 'only one of log_transform_input and arcsinh_transform_input can be True'

        if self.return_cosmo:
            assert cosmo_params_path is not None, "cosmo_params_path must be provided if return_cosmo is True"
            raw_params = np.loadtxt(cosmo_params_path)
            mean = raw_params.mean(axis=0)
            std = raw_params.std(axis=0)
            self.cosmo_params = torch.tensor((raw_params - mean) / std, dtype=torch.float32)
        
        self.input_files = [
            os.path.join(root, get_filepath(sample_no, input_type))
            for sample_no in idx_list
        ]
        self.target_files = [
            os.path.join(root, get_filepath(sample_no, target_type))
            for sample_no in idx_list
        ]
        
        # sanity check
        for input_file, target_file in zip(self.input_files, self.target_files):
            assert os.path.exists(input_file), f'{input_file} does not exist'
            assert os.path.exists(target_file), f'{target_file} does not exist'
        
        self.n_samples = len(idx_list)
        
        if augment: 
            self.augment = CosmoAugment3D()
        else:
            self.augment = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        inputs = np.load(self.input_files[idx])
        
        # LOG TRANSFORM THE INPUT
        # We shift by +1.0 because delta min is -1. 
        # Adding epsilon prevents log(0).
        # cond_z0 becomes more Gaussian-like.
        if self.log_transform_input:
            # The halo catalog is an overdensity field, similar to z=0 density fields.
            # Using the same transformation for consistency.
            inputs = np.log10(inputs + 1 + 1e-5)

            if self.normalize:
                mean, std = stats_dict[self.input_type]
                inputs = (inputs - mean) / std

        elif self.arcsinh_transform_input:
            std = raw_stats[self.input_type][1]
            inputs = np.arcsinh(inputs/std)
            
            if self.normalize:
                mean, std = arcsinh_stats[self.input_type]
                inputs = (inputs - mean) / std
            
        elif self.normalize:
            inputs = (inputs - inputs.mean()) / inputs.std()
        
        inputs = torch.from_numpy(inputs)
        if inputs.ndim ==3: inputs = inputs.unsqueeze(0)
        
        label  = np.load(self.target_files[idx])
        if self.local_normalize_target:
            label = (label - label.mean()) / label.std() 
        elif self.min_max_target:
            label = (label - label.min()) / (label.max() - label.min())
        elif self.normalize:
            mean, std = stats_dict[self.target_type]
            label = (label - mean) / std 
            
        label = torch.from_numpy(label)
        if label.ndim == 3: label = label.unsqueeze(0)
        
        if self.augment:
            inputs, label = self.augment(inputs, label)
        
        if self.return_cosmo:
            sample_no = self.idx_list[idx]
            theta = self.cosmo_params[sample_no]
            return inputs, label, theta

        return inputs, label
    

def calculate_global_stats(root, idx_list, data_type, apply_log=False):
    """
    Calculates global mean and std incrementally.
    """
    # Accumulators
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    
    # Get file list
    files = [os.path.join(root, get_filepath(i, data_type)) for i in idx_list]
    
    print(f"Computing statistics for {len(files)} files...")
    
    for fpath in tqdm(files):
        if not os.path.exists(fpath):
            print(f"Warning: {fpath} not found.")
            continue
            
        # Load one file at a time
        data = np.load(fpath).astype(np.float64) # Use float64 for precision during sum
        if apply_log:
            data = np.log10(data + 1 + 1e-5)

        # Accumulate
        total_sum += np.sum(data)
        total_sq_sum += np.sum(data ** 2)
        total_count += data.size
        
    # Final Calculation
    global_mean = total_sum / total_count
    
    # Variance = E[X^2] - (E[X])^2
    global_variance = (total_sq_sum / total_count) - (global_mean ** 2)
    global_std = np.sqrt(global_variance)
    
    return global_mean, global_std

class CosmoAugment3D:
    """
    Applies consistent stochastic augmentations to a pair of 3D fields (x, y).
    Valid for cosmological simulations with Periodic Boundary Conditions.
    """
    def __init__(self, p_flip=0.0, p_rot=0.0, p_trans=0.5, max_trans=16):
        # Flips and Rotations MUST be 0 when using deterministic Positional Encodings,
        # otherwise the physics orientation decouples from the PE gradient vectors!
        self.p_flip = p_flip
        self.p_rot = p_rot
        self.p_trans = p_trans
        self.max_trans = max_trans  # Max pixels to shift (e.g., 16 on a 128 grid)

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        """
        x, y: Tensors of shape [C, D, H, W]
        """
        # 1. Random Flips (Mirroring)
        # We check each axis (D, H, W) independently
        dims = [1, 2, 3] # [D, H, W] assuming shape [C, D, H, W]
        
        if self.p_flip > 0:
            for dim in dims:
                if random.random() < self.p_flip:
                    x = torch.flip(x, [dim])
                    y = torch.flip(y, [dim])

        # 2. Random 90-Degree Rotations
        # There are 3 principal axes (0-1, 0-2, 1-2 planes)
        if self.p_rot > 0 and random.random() < self.p_rot:
            # Pick a random plane to rotate 90 degrees in
            # (1, 2) = Rotate around W axis
            # (1, 3) = Rotate around H axis
            # (2, 3) = Rotate around D axis
            rot_dims = random.choice([(1, 2), (1, 3), (2, 3)])
            
            # Pick 1, 2, or 3 rotations (90, 180, 270 deg)
            k = random.randint(1, 3)
            
            x = torch.rot90(x, k, rot_dims)
            y = torch.rot90(y, k, rot_dims)

        # 3. Random Cyclic Shifts (Translation)
        # Only valid for Periodic Boundary Conditions!
        if self.p_trans > 0 and random.random() < self.p_trans:
            shifts = [random.randint(-self.max_trans, self.max_trans) for _ in range(3)]
            
            # torch.roll handles circular shifting automatically
            x = torch.roll(x, shifts=shifts, dims=(1, 2, 3))
            y = torch.roll(y, shifts=shifts, dims=(1, 2, 3))

        return x, y

class PatchAugment3D:
    """
    Stochastic augmentations tailored for pre-extracted sub-patches.
    Note: Translations (torch.roll) are intentionally omitted because 
    local patches do not have periodic boundary conditions.
    """
    def __init__(self, p_flip=0.0, p_rot=0.0):
        # Flips and Rotations MUST be 0 when using deterministic Positional Encodings,
        # otherwise the physics orientation decouples from the PE gradient vectors!
        self.p_flip = 0.0
        self.p_rot = 0.0

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        dims = [1, 2, 3] # [D, H, W] assuming shape [C, D, H, W]
        
        # 1. Random Flips
        if self.p_flip > 0:
            for dim in dims:
                if random.random() < self.p_flip:
                    x = torch.flip(x, [dim])
                    y = torch.flip(y, [dim])

        # 2. Random 90-Degree Rotations
        if self.p_rot > 0 and random.random() < self.p_rot:
            rot_dims = random.choice([(1, 2), (1, 3), (2, 3)])
            k = random.randint(1, 3)
            
            x = torch.rot90(x, k, rot_dims)
            y = torch.rot90(y, k, rot_dims)

        return x, y

class SingleAugment3D:
    """
    Applies consistent stochastic augmentations to a pair of 3D fields (x, y).
    Valid for cosmological simulations with Periodic Boundary Conditions.
    """
    def __init__(self, p_flip=0.0, p_rot=0.0, p_trans=0.5, max_trans=16):
        # Flips and Rotations MUST be 0 when using deterministic Positional Encodings,
        # otherwise the physics orientation decouples from the PE gradient vectors!
        self.p_flip = p_flip
        self.p_rot = p_rot
        self.p_trans = p_trans
        self.max_trans = max_trans  # Max pixels to shift (e.g., 16 on a 128 grid)
    
    def __call__(self, x: torch.Tensor):
        """
        x: Tensor of shape [C, D, H, W]
        """
        # 1. Random Flips (Mirroring)
        # We check each axis (D, H, W) independently
        dims = [1, 2, 3] # [D, H, W] assuming shape [C, D, H, W]
        
        if self.p_flip > 0:
            for dim in dims:
                if random.random() < self.p_flip:
                    x = torch.flip(x, [dim])

        # 2. Random 90-Degree Rotations
        # There are 3 principal axes (0-1, 0-2, 1-2 planes)
        if self.p_rot > 0 and random.random() < self.p_rot:
            # Pick a random plane to rotate 90 degrees in
            # (1, 2) = Rotate around W axis
            # (1, 3) = Rotate around H axis
            # (2, 3) = Rotate around D axis
            rot_dims = random.choice([(1, 2), (1, 3), (2, 3)])
            
            # Pick 1, 2, or 3 rotations (90, 180, 270 deg)
            k = random.randint(1, 3)
            
            x = torch.rot90(x, k, rot_dims)

        # 3. Random Cyclic Shifts (Translation)
        # Only valid for Periodic Boundary Conditions!
        if self.p_trans > 0 and random.random() < self.p_trans:
            shifts = [random.randint(-self.max_trans, self.max_trans) for _ in range(3)]
            
            # torch.roll handles circular shifting automatically
            x = torch.roll(x, shifts=shifts, dims=(1, 2, 3))

        return x


class SingleDataset(Dataset):
    def __init__(
        self, root, idx_list,
        input_type='standard_z0_128',
        normalize=True, augment=True,
        log_transform_input=True,
        return_cosmo=False,
        cosmo_params_path=None
    ):
        self.root = root
        self.normalize = normalize
        self.input_type = input_type
        self.log_transform_input = log_transform_input # mainly for the z0 data
        self.idx_list = list(idx_list)
        self.return_cosmo = return_cosmo

        if self.return_cosmo:
            assert cosmo_params_path is not None, "cosmo_params_path must be provided if return_cosmo is True"
            raw_params = np.loadtxt(cosmo_params_path)
            mean = raw_params.mean(axis=0)
            std = raw_params.std(axis=0)
            self.cosmo_params = torch.tensor((raw_params - mean) / std, dtype=torch.float32)
                
        self.input_files = [
            os.path.join(root, get_filepath(sample_no, input_type))
            for sample_no in idx_list
        ]
        
        # sanity check
        for input_file in self.input_files:
            assert os.path.exists(input_file), f'{input_file} does not exist'
        
        self.n_samples = len(idx_list)
        
        if augment: 
            self.augment = SingleAugment3D(
                p_flip=0.25, p_rot=0.25, p_trans=0, max_trans=0
            )
        else:
            self.augment = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        inputs = np.load(self.input_files[idx])
        if self.log_transform_input:
            # The halo catalog is an overdensity field, similar to z=0 density fields.
            # Using the same transformation for consistency.
            inputs = np.log10(inputs + 1 + 1e-5)
            
        if self.normalize:
            mean, std = stats_dict[self.input_type]
            inputs = (inputs - mean) / std
        
        inputs = torch.from_numpy(inputs).unsqueeze(0)
        
        if self.augment:
            inputs = self.augment(inputs)
        
        if self.return_cosmo:
            sample_no = self.idx_list[idx]
            theta = self.cosmo_params[sample_no]
            return inputs, theta

        return inputs

class CosmoFOLDCollator:
    """
    Faithful to CosmoFOLD Algorithm 1: extract ALL non-overlapping fixed-grid
    patches from each volume. Each volume contributes (D/p)*(H/p)*(W/p) patches.

    SimulationDataset returns (inputs [C, D, H, W], label [C, D, H, W]).
    This collator ignores the DataLoader batch_size and instead uses
    patches_per_volume to control GPU batch size.

    Returns:
        x           [B, C_in,  p, p, p]
        y           [B, C_out, p, p, p]
        vol_shape   [B, 3]
        patch_starts [B, 3]
    where B = len(samples) * patches_per_volume, randomly subsampled to
    batch_size if specified.
    """

    def __init__(
        self,
        patch_size:  int = 64,
        batch_size:  int = 32,       # how many patches to actually return
        sample_all:  bool = False,  # True = return all patches (inference mode)
    ):
        self.p           = patch_size
        self.batch_size  = batch_size
        self.sample_all  = sample_all

    def __call__(self, batch):
        p = self.p
        all_possible_patches = []

        # 1. Compute all valid coordinates across the entire batch (virtually no memory cost)
        for b_idx, item in enumerate(batch):
            inputs = item[0]
            _, D, H, W = inputs.shape
            for d in range(0, D, p):
                for h in range(0, H, p):
                    for w in range(0, W, p):
                        d0 = min(d, D - p)
                        h0 = min(h, H - p)
                        w0 = min(w, W - p)
                        all_possible_patches.append((b_idx, d0, h0, w0))

        # 2. Randomly subsample the coordinates FIRST
        if self.sample_all:
            selected_patches = all_possible_patches
        else:
            n = len(all_possible_patches)
            indices = torch.randperm(n)[:self.batch_size].tolist()
            selected_patches = [all_possible_patches[i] for i in indices]

        # 3. Only slice and clone the explicitly requested patches
        all_x, all_y, all_starts = [], [], []
        all_theta = []
        for b_idx, d0, h0, w0 in selected_patches:
            item = batch[b_idx]
            inputs = item[0]
            label = item[1]
            all_x.append(inputs[:, d0:d0+p, h0:h0+p, w0:w0+p].clone())
            all_y.append(label[:,  d0:d0+p, h0:h0+p, w0:w0+p].clone())
            all_starts.append(torch.tensor([d0, h0, w0], dtype=torch.long))
            if len(item) > 2:
                all_theta.append(item[2])

        if len(all_theta) > 0:
            return (
                torch.stack(all_x, dim=0),
                torch.stack(all_y, dim=0),
                torch.stack(all_starts, dim=0),
                torch.stack(all_theta, dim=0)
            )

        return (
            torch.stack(all_x, dim=0),
            torch.stack(all_y, dim=0),
            torch.stack(all_starts, dim=0),
        )

class CachedPatchDataset(Dataset):
    """
    Loads pre-extracted 3D patches saved as .npz files.
    Completely eliminates the memory bottleneck of loading massive volumes.
    Returns: inputs [1, p, p, p], label [1, p, p, p], starts [3]
    """
    def __init__(
        self, patch_dir, idx_list=None,
        input_type='standard_z0_128', target_type='standard_ic_128',
        normalize=True, log_transform_input=True,
        return_cosmo=False, cosmo_params_path=None
    ):
        self.patch_dir = patch_dir
        self.normalize = normalize
        self.input_type = input_type
        self.target_type = target_type
        self.log_transform_input = log_transform_input
        
        self.files = []
        self.sample_nos = []
        if idx_list is not None:
            for idx in idx_list:
                sim_dir = os.path.join(patch_dir, f"sim_{idx:04d}")
                if os.path.exists(sim_dir):
                    for f in sorted(os.listdir(sim_dir)):
                        if f.endswith('.npz'):
                            self.files.append(os.path.join(sim_dir, f))
                            self.sample_nos.append(idx)
        else:
            for root_dir, _, fnames in os.walk(patch_dir):
                sim_name = os.path.basename(root_dir)
                if sim_name.startswith("sim_"):
                    try:
                        sim_idx = int(sim_name.split("_")[1])
                    except ValueError:
                        sim_idx = 0
                else:
                    sim_idx = 0
                for f in sorted(fnames):
                    if f.endswith('.npz'):
                        self.files.append(os.path.join(root_dir, f))
                        self.sample_nos.append(sim_idx)
                    
        self.n_samples = len(self.files)

        self.return_cosmo = return_cosmo
        if self.return_cosmo:
            assert cosmo_params_path is not None, "cosmo_params_path must be provided if return_cosmo is True"
            raw_params = np.loadtxt(cosmo_params_path)
            mean = raw_params.mean(axis=0)
            std = raw_params.std(axis=0)
            self.cosmo_params = torch.tensor((raw_params - mean) / std, dtype=torch.float32)
        
        # --- FIX: Prevent silent DDP deadlocks caused by NFS sync delays ---
        if self.n_samples == 0:
            raise RuntimeError(f"FATAL: No cached patches found in {patch_dir}! (NFS caching delay or wrong path)")
        print(f"Found {self.n_samples} cached patches in {patch_dir}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        try:
            data = np.load(self.files[idx])
            inputs = data['input']
            label = data['target']
            starts = data['starts']
        except Exception as e:
            print(f"WARNING: Skipping corrupted file {self.files[idx]} due to error: {e}")
            # raise e
            new_idx = random.randint(0, self.n_samples - 1)
            return self.__getitem__(new_idx)
        
        if self.log_transform_input:
            # The halo catalog is an overdensity field, similar to z=0 density fields.
            # Using the same transformation for consistency.
            inputs = np.log10(inputs + 1 + 1e-5)
            
        if self.normalize:
            mean_in, std_in = stats_dict[self.input_type]
            inputs = (inputs - mean_in) / std_in
            
            mean_tar, std_tar = stats_dict[self.target_type]
            label = (label - mean_tar) / std_tar 
            
        inputs = torch.from_numpy(inputs)
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)  # [1, p, p, p]
            
        label = torch.from_numpy(label)
        if label.ndim == 3:
            label = label.unsqueeze(0)    # [1, p, p, p]
            
        starts = torch.from_numpy(starts) # [3]
            
        if self.return_cosmo:
            sample_no = self.sample_nos[idx]
            theta = self.cosmo_params[sample_no]
            return inputs, label, starts, theta

        return inputs, label, starts

def get_dataloaders(
    batch_size,
    num_workers,
    idx_train,
    idx_val,
    root,
    normalize=True,
    input_type='standard_z0_128',
    target_type='standard_ic_128',
    augment=True,
    log_transform_input=True,
    collate_fn=None,
    use_cached=False,
    patch_dir=None,
    return_cosmo=False,
    cosmo_params_path=None,
    use_cached_wavelet=False,
    patch_size=None,
    local_normalize_target=False,
    min_max_target= False,
    arcsinh_transform_input=False
):
    print(f"Using {len(idx_train)} training points and {len(idx_val)} validation points.")
    if use_cached and patch_dir is not None:
        if augment:
            print('Cached patches can not be augmented, since that will break consistency with the large image')
        train_data = CachedPatchDataset(
            patch_dir=patch_dir, idx_list=idx_train,
            input_type=input_type, target_type=target_type,
            normalize=normalize,
            log_transform_input=log_transform_input,
            return_cosmo=return_cosmo,
            cosmo_params_path=cosmo_params_path
        )
        val_data = CachedPatchDataset(
            patch_dir=patch_dir, idx_list=idx_val,
            input_type=input_type, target_type=target_type,
            normalize=normalize,
            log_transform_input=log_transform_input,
            return_cosmo=return_cosmo,
            cosmo_params_path=cosmo_params_path
        )
    elif use_cached_wavelet and patch_dir is not None:

        from dataset.cached_wavelet_dataset import CachedWaveletDataset
        print('Using CachedWaveletDataset')
        
        train_data = CachedWaveletDataset(
            cache_dir=patch_dir, idx_list=idx_train,
            patch_size=patch_size,
            augment=augment,
            return_cosmo=return_cosmo,
            cosmo_params_path=cosmo_params_path
        )
        val_data = CachedWaveletDataset(
            cache_dir=patch_dir, idx_list=idx_val,
            patch_size=patch_size,
            augment=False,
            return_cosmo=return_cosmo,
            cosmo_params_path=cosmo_params_path
        )

        
    else:
        train_data = SimulationDataset(
            root=root, idx_list=idx_train,
            input_type=input_type, target_type=target_type,
            augment=augment, normalize=normalize,
            log_transform_input=log_transform_input,
            return_cosmo=return_cosmo,
            cosmo_params_path=cosmo_params_path,
            local_normalize_target=local_normalize_target,
            min_max_target=min_max_target,
            arcsinh_transform_input=arcsinh_transform_input
        )
        val_data = SimulationDataset(
            root=root, idx_list=idx_val,
            input_type=input_type, target_type=target_type,
            augment=False, normalize=normalize,
            log_transform_input=log_transform_input,
            return_cosmo=return_cosmo,
            cosmo_params_path=cosmo_params_path,
            local_normalize_target=local_normalize_target,
            min_max_target=min_max_target,
            arcsinh_transform_input=arcsinh_transform_input
        )
        
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor= 2 if num_workers > 2 else None,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor= 2 if num_workers > 2 else None,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    return train_loader, val_loader
