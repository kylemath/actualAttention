#!/usr/bin/env python3
"""
Run all experiments in parallel for fast iteration.

Designed for M5 Studio with 16 cores and 128GB RAM.
Runs all 18 experiments (3 archs × 2 variants × 3 seeds) simultaneously.
"""

import argparse
import multiprocessing as mp
import time
from pathlib import Path
from run_experiments import run_single_experiment, PATHS

def run_single_config(args):
    """Run a single experiment configuration."""
    arch, task, mode, bio, seed = args
    try:
        print(f"[{arch}|{'bio' if bio else 'base'}|seed{seed}] Starting...")
        start = time.time()
        result = run_single_experiment(arch, task, mode, bio, seed)
        elapsed = time.time() - start
        print(f"[{arch}|{'bio' if bio else 'base'}|seed{seed}] Done in {elapsed/60:.1f}m")
        return result
    except Exception as e:
        print(f"[{arch}|{'bio' if bio else 'base'}|seed{seed}] ERROR: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Run experiments in parallel')
    parser.add_argument('--mode', type=str, default='parallel', 
                        help='Experiment mode (parallel, fast, cycle)')
    parser.add_argument('--tasks', nargs='+', default=['copy'])
    parser.add_argument('--archs', nargs='+', default=['rnn', 'lstm', 'transformer'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: # of CPUs)')
    parser.add_argument('--output-dir', type=str, default=PATHS['data'])
    args = parser.parse_args()

    # Build list of all experiment configurations
    configs = []
    for task in args.tasks:
        for arch in args.archs:
            for bio in [False, True]:
                for seed in args.seeds:
                    configs.append((arch, task, args.mode, bio, seed))
    
    print(f"Running {len(configs)} experiments in parallel...")
    print(f"Mode: {args.mode}")
    print(f"Workers: {args.workers or mp.cpu_count()}")
    print(f"Output: {args.output_dir}")
    print("-" * 60)
    
    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run all experiments in parallel
    start_time = time.time()
    
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(run_single_config, configs)
    
    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r is not None)
    
    print("-" * 60)
    print(f"Completed: {successful}/{len(configs)} experiments")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"Average per experiment: {elapsed/len(configs)/60:.1f} minutes")
    
    # Save results
    from utils import save_results
    for i, result in enumerate(results):
        if result is not None:
            arch, task, mode, bio, seed = configs[i]
            tag = f"{args.mode}_{arch}_{task}_{'bio' if bio else 'base'}_seed{seed}"
            save_results(result, f"{tag}.pkl", base_dir=args.output_dir)
    
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == '__main__':
    # Set start method for macOS
    mp.set_start_method('spawn', force=True)
    main()
