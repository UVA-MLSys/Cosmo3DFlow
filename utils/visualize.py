import matplotlib.pyplot as plt
from dataset.dataset import stats_dict
import Pk_library as PKL
from utils.metrics import pspec, cross_pspec
import numpy as np
import torch, os

def initialize_plt():
    plt.rcParams.update({
        'font.size': 16,          # Default text size
        'axes.titlesize': 18,     # Title of each subplot (e.g., "Input (Max Proj)")
        'axes.labelsize': 16,     # x and y labels
        'xtick.labelsize': 14,    # Tick numbers on x-axis
        'ytick.labelsize': 14,    # Tick numbers on y-axis
        'legend.fontsize': 16,    # Legend text
        'figure.titlesize': 20,   # Super title (e.g., "Epoch 10")
        'figure.figsize': (12, 14) # Set default figure size if you want
    })

# def visualize(y, pred, output_dir, epoch, save_locally=True):
#         # Visualize center slices of 3D volumes
#         fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
#         # Original - 3 orthogonal slices
#         mid_d, mid_h, mid_w = y.shape[2]//2, y.shape[3]//2, y.shape[4]//2
#         axes[0, 0].imshow(y[0, 0, mid_d, :, :].cpu().numpy(), cmap='viridis')
#         axes[0, 0].set_title("Original (D slice)")
#         axes[0, 1].imshow(y[0, 0, :, mid_h, :].cpu().numpy(), cmap='viridis')
#         axes[0, 1].set_title("Original (H slice)")
#         axes[0, 2].imshow(y[0, 0, :, :, mid_w].cpu().numpy(), cmap='viridis')
#         axes[0, 2].set_title("Original (W slice)")
        
#         # Reconstructed - 3 orthogonal slices
#         axes[1, 0].imshow(pred[0, 0, mid_d, :, :].cpu().numpy(), cmap='viridis')
#         axes[1, 0].set_title("Reconstructed (D slice)")
#         axes[1, 1].imshow(pred[0, 0, :, mid_h, :].cpu().numpy(), cmap='viridis')
#         axes[1, 1].set_title("Reconstructed (H slice)")
#         axes[1, 2].imshow(pred[0, 0, :, :, mid_w].cpu().numpy(), cmap='viridis')
#         axes[1, 2].set_title("Reconstructed (W slice)")
        
#         plt.tight_layout()
#         if save_locally:
#             plt.savefig(f'{output_dir}/{epoch}.jpg', bbox_inches="tight", dpi=300)
#             plt.close(fig)
#         else:
#             plt.show()
            
def visualize(
    y, pred, output_dir, epoch, 
    save_locally=True, target_type='standard_ic_128',
    boxsize=1000.0, kmax=0.4
):
    y = y.cpu().squeeze().numpy()
    pred = pred.cpu().squeeze().numpy()
    
    # This par
    mean, std = stats_dict[target_type]
    # Images are normalized, so we need to denormalize them
    y1 = y * std + mean
    pred1 = pred * std + mean

    if 'ic' in target_type:
        truth_mas = 'CIC'
    elif 'z0' in target_type:
        truth_mas = 'PCS'
    else:
        truth_mas = 'None'

    Pk_orig, k_orig = pspec(y1, boxsize=boxsize, kmax=kmax, mas=truth_mas)
    Pk_recon, k_recon = pspec(pred1, boxsize=boxsize, kmax=kmax, mas='None')
    
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(3, 3 , height_ratios=[1, 1, 0.8])

    # Original - 3 orthogonal slices
    mid_d, mid_h, mid_w = y.shape[0]//2, y.shape[1]//2, y.shape[2]//2
    axes_0_0 = fig.add_subplot(gs[0, 0])
    axes_0_0.imshow(y[mid_d, :, :], cmap='viridis')
    axes_0_0.set_title("Original (D slice)")

    axes_0_1 = fig.add_subplot(gs[0, 1])
    axes_0_1.imshow(y[:, mid_h, :], cmap='viridis')
    axes_0_1.set_title("Original (H slice)")

    axes_0_2 = fig.add_subplot(gs[0, 2])
    axes_0_2.imshow(y[:, :, mid_w], cmap='viridis')
    axes_0_2.set_title("Original (W slice)")

    # Reconstructed - 3 orthogonal slices
    axes_1_0 = fig.add_subplot(gs[1, 0])
    axes_1_0.imshow(pred[mid_d, :, :], cmap='viridis')
    axes_1_0.set_title("Reconstructed (D slice)")

    axes_1_1 = fig.add_subplot(gs[1, 1])
    axes_1_1.imshow(pred[:, mid_h, :], cmap='viridis')
    axes_1_1.set_title("Reconstructed (H slice)")

    axes_1_2 = fig.add_subplot(gs[1, 2])
    axes_1_2.imshow(pred[:, :, mid_w], cmap='viridis')
    axes_1_2.set_title("Reconstructed (W slice)")

    axes_power = fig.add_subplot(gs[2, :])
    axes_power.loglog(k_orig, Pk_orig, label="Original")
    axes_power.loglog(k_recon, Pk_recon, label="Reconstruction")

    axes_power.set_xlabel('k [h/Mpc]')
    axes_power.set_ylabel('P(k) [Mpc/h]^3')
    axes_power.set_title('Power Spectrum Comparison')
    axes_power.legend()

    plt.tight_layout()
    
    if save_locally:
        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/{epoch}.jpg', bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()
        
def visualize_pspec(
    model, x, y, output_dir, epoch, 
    save_locally=True, 
    target_type='standard_ic_128',
    boxsize=1000.0,
    kmax=0.4
):
    h = 3
    w = 4
    mean, std = stats_dict[target_type]

    truth = y * std + mean
    
    if type(truth) == torch.Tensor:
        truth = truth.squeeze().detach().cpu().numpy()
    
    recons = []
    n_samples = 3

    with torch.no_grad():
        for _ in range(n_samples):
            latent = model.encoder(x)

            # Available solvers are 'euler', 'rk4', and 'dopri5'
            recon = model.decoder.predict(
                x0=torch.randn_like(y),
                h=latent,
                n_sampling_steps=50,
                solver='euler',
            )
            recons.append(recon.cpu().numpy())

    samples = np.array(recons).squeeze() # -1,Nside,Nside,Nside
    # unnormalized the samples as well
    samples = samples * std + mean
    
    if 'ic' in target_type:
        truth_mas = 'CIC'
    elif 'z0' in target_type:
        truth_mas = 'PCS'
    else:
        truth_mas = 'None'

    truth_pspec, truth_k = pspec(truth, boxsize=boxsize, kmax=1, mas=truth_mas)

    samples_crosspspec = []
    samples_pspec = []
    samples_k = []

    for i, sample in enumerate(samples):
        ps, k = pspec(sample, mas='None')
        ps_cross, _ = cross_pspec(sample, truth, boxsize=boxsize, kmax=kmax, mas=['None', truth_mas])
        samples_crosspspec.append(ps_cross)
        samples_pspec.append(ps)
        samples_k.append(k)


    samples_pspec = np.array(samples_pspec)
    samples_k = np.array(samples_k)
    mean_pspec = np.mean(samples_pspec, axis=0)
    std_pspec = np.std(samples_pspec, axis=0)

    samples_crosspspec = np.array(samples_crosspspec)
    mean_crossps = np.mean(samples_crosspspec, axis=0)
    std_crossps = np.std(samples_crosspspec, axis=0)
    
    # calculate transfer
    tf_set = []
    for i in range(samples_pspec.shape[0]):
        tf_set.append(np.sqrt((samples_pspec[i]+1e-6)/(truth_pspec+1e-6)))
    tf_set = np.array(tf_set)
    mean_tf = np.mean(tf_set, axis=0)
    std_tf = np.std(tf_set, axis=0)
    samples_cross_k = np.array(samples_cross_k)
    
    
    from matplotlib.ticker import FormatStrFormatter
    import matplotlib.ticker as ticker

    fig, axs = plt.subplots(
        3, sharex=True, sharey=False, height_ratios=[2, 1, 1]
    )
    # Plot power spectra of truth vs generated samples
    fig.set_size_inches((w*2, h*3)) 
    axs[0].plot(samples_k[-1], mean_pspec, color='#82A8D1', label='Inferred')
    axs[0].fill_between(samples_k[-1], mean_pspec - 2*std_pspec, mean_pspec+2*std_pspec, alpha=0.5, color='#82A8D1')
    axs[0].plot(truth_k, truth_pspec, color='k', ls='--', lw=1, label='Truth')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].tick_params(axis='x', which='both',length=0)
    #if np.sum(cosmo_idx == np.array([50,70,80,40])) == 0:
    axs[0].set_ylabel(r"$P(k)$")
    
    axs[0].set_xlim(left=samples_k[-1,0])

    # Plot cross-correlation of true vs samples
    axs[1].plot(samples_cross_k[-1], mean_crossps, color='#82A8D1')
    axs[1].fill_between(samples_cross_k[-1], mean_crossps - 2*std_crossps, mean_crossps+2*std_crossps, alpha=0.5, color='#82A8D1')
    axs[1].axhline(1.0, color='k', ls='--', lw=1)
    axs[1].set_xscale('log')
    axs[1].tick_params(axis='x', which='both',length=0)
    axs[1].set_ylabel(r"$C(k)$")
    axs[1].set_xlim(left=samples_cross_k[-1,0])

    # Plot transfer function of sample
    axs[2].plot(samples_k[-1], mean_tf, color='#82A8D1')
    axs[2].fill_between(samples_k[-1], mean_tf - 2*std_tf, mean_tf+2*std_tf, alpha=0.5, color='#82A8D1')
    axs[2].axhline(1.0, color='k', ls='--', lw=1)
    axs[2].set_xscale('log')
    axs[2].set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    axs[2].set_ylabel(r"$T(k)$")
    axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    # axs[2].set_ylim(bottom=0.95,top=1.1)

    if np.max(mean_tf) < 1.0:
        axs[2].set_ylim(top=1.005)
        
    axs[2].set_xlim(left=samples_k[-1,0])  
    axs[2].xaxis.set_major_locator(ticker.MultipleLocator(1e-1))

    plt.subplots_adjust(hspace=0)
    
    if save_locally:
        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'pspec.jpg'), bbox_inches='tight', dpi=200)
        plt.close(fig)
    else:
        plt.show()