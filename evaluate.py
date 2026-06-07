# module load gcc openmpi miniforge

import os

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import Pk_library as PKL

from tqdm import tqdm

import scipy.stats as stats
from typing import Dict, List, Tuple
import json
from utils.metrics import ValidationSuite, ValidationMetrics
import argparse
from dataset.dataset import get_filepath, stats_dict

parser = argparse.ArgumentParser(
    description='Evaluate generated samples', 
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--root', type=str, default='../Datasets')
parser.add_argument('--model_folder', type=str, default='./run/1')
parser.add_argument('--boxsize', type=float, default=1000)
parser.add_argument('--target_type', type=str, default='standard_ic_128')
parser.add_argument('--kmax', type=float, default=None)
parser.add_argument('--start', type=int, default=1900)
parser.add_argument('--end', type=int, default=2000)
parser.add_argument('--normalize_for_vrmse', action='store_true', help='Globally normalize samples and truth for VRMSE metric calculation.')
# The --unnormalize_sample flag is removed as sample.py now consistently saves samples in physical units.
parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel workers for evaluation')
parser.add_argument('--threads', type=int, default=5, help='Number of threads for power spectrum computation')
parser.add_argument('--compute_diversity_metrics', action='store_true', help='whether to compute diversity and calibration metrics')

args = parser.parse_args()

root = args.root
model_folder = args.model_folder

global_mean, global_std = stats_dict[args.target_type]

val_suite = ValidationSuite(
    boxsize=args.boxsize, kmax=args.kmax, threads=args.threads, truth_mas='None',
    compute_diversity_metrics=args.compute_diversity_metrics
)

def evaluate_single_sample(sample_no, args, model_folder, root, global_mean, global_std, metrics_computer):
    try:
        sample_folder = os.path.join(model_folder, 'samples', str(sample_no))
        
        sample_path = os.path.join(sample_folder, 'sample.npy')
        if os.path.exists(sample_path):
            samples_phys = np.load(sample_path).squeeze()
            if samples_phys.ndim == 3:
                samples_phys = np.expand_dims(samples_phys, axis=0)
        else:
            sample_files = [f for f in os.listdir(sample_folder) if f.startswith('sample_') and f.endswith('.npy')]
            sample_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            samples_phys = np.stack([np.load(os.path.join(sample_folder, f)).squeeze() for f in sample_files])
            
        truth_phys = np.load(os.path.join(root, get_filepath(sample_no, args.target_type)))

        # VRMSE is the only metric that requires globally normalized fields.
        # Other metrics like P(k), T(k), C(k) operate on physical fields.
        if args.normalize_for_vrmse:
            samples_global = (samples_phys - global_mean) / global_std
            truth_global = (truth_phys - global_mean) / global_std
        else: 
            samples_global = None 
            truth_global = None 

        example_metrics = metrics_computer.compute_all_metrics(
            samples_phys, truth_phys, samples_global, truth_global
        )
        return sample_no, example_metrics
    except Exception as e:
        return sample_no, e

total_samples = args.end - args.start
if args.num_workers > 1:
    print(f"Evaluating using {args.num_workers} workers...")
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(evaluate_single_sample, sample_no, args, model_folder, root, global_mean, global_std, val_suite.metrics_computer): sample_no
            for sample_no in range(args.start, args.end)
        }
        for i, future in tqdm(enumerate(as_completed(futures)), total=len(futures)):
            sample_no, result = future.result()
            if isinstance(result, Exception):
                print(f"Error evaluating sample {sample_no}: {result}")
                continue
            val_suite.add_example_metrics(result, sample_no)
            if (i + 1) % 10 == 0 and (i + 1) < total_samples:
                print(f"After {i+1} examples: {val_suite._get_current_stats()}")
            results = val_suite._finalize_stats()
            val_suite.save_results(results, os.path.join(model_folder, 'results.json'))
else:
    for i, sample_no in tqdm(enumerate(range(args.start, args.end)), total=total_samples):
        _, result = evaluate_single_sample(sample_no, args, model_folder, root, global_mean, global_std, val_suite.metrics_computer)
        if isinstance(result, Exception):
            print(f"Error evaluating sample {sample_no}: {result}")
            continue
        val_suite.add_example_metrics(result, sample_no)
        if (i + 1) % 10 == 0 and (i + 1) < total_samples:
            print(f"After {i+1} examples: {val_suite._get_current_stats()}")
        results = val_suite._finalize_stats()
        val_suite.save_results(results, os.path.join(model_folder, 'results.json'))

results = val_suite._finalize_stats()

val_suite.print_summary(results)
# The final save is now redundant as it's done inside the loop.