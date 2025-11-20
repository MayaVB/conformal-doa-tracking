import csv
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy and PyTorch.
    Also configures cuDNN for deterministic behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # For CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_multi_scenario_results(
    results: List[Dict[str, Any]],
    args: Any,
    csv_file: Optional[str],
    selected_checkpoint: Optional[str],
) -> None:
    """
    Write logging info and a results.txt summary for multi-scenario testing.

    Args:
        results:
            List of dictionaries containing test results for each scenario.
            Expected keys include:
                'unet_mae', 'filter_mae', 'doa_mse', 'cp_coverage',
                'checkpoints_dir', 'rtf_esti_method', 'feature_op',
                'num_classes', 'lambda_spacing', 'lambda_size',
                'lambda_span', 'threshold', 'scenario'.
        args:
            Argument namespace containing at least 'test_data_path'.
        csv_file:
            Path to the exported CSV file (from export_results_to_csv).
        selected_checkpoint:
            Path to the loaded checkpoint file, or None.
    """
    if not results:
        logging.warning("write_multi_scenario_results called with empty results list.")
        return

    # Compute overall statistics across all scenarios
    overall_unet_mae = np.mean([r["unet_mae"] for r in results])
    overall_filter_mae = np.mean([r["filter_mae"] for r in results])
    overall_mse = np.mean([r["doa_mse"] for r in results])
    overall_coverage = np.mean([r["cp_coverage"] for r in results])

    # Configuration from first result (all scenarios share config)
    first_result = results[0]

    # Build output path
    path = Path(args.test_data_path)
    subfolder = path.parent.name
    plots_dir = Path("plots") / subfolder
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_file = plots_dir / "results.txt"
    with summary_file.open("w") as f:
        f.write("#######################Configuration:\n")
        f.write(f"Data path: {args.test_data_path}\n")
        f.write(f'Checkpoints dir: {first_result["checkpoints_dir"]}\n')
        if selected_checkpoint:
            f.write(f"Checkpoint file: {selected_checkpoint}\n")
        f.write(f'RTF estimation method: {first_result["rtf_esti_method"]}\n')
        f.write(f'Feature operation: {first_result["feature_op"]}\n')
        f.write(f'Number of classes: {first_result["num_classes"]}\n')
        f.write(
            f'CP penalty for spacing (lambda): '
            f'{first_result["lambda_spacing"]:.2f}\n'
        )
        f.write(
            f'CP penalty for size (lambda): '
            f'{first_result["lambda_size"]:.2f}\n'
        )
        f.write(
            f'CP penalty for span (lambda): '
            f'{first_result["lambda_span"]:.2f}\n'
        )
        f.write(f'CP threshold: {first_result["threshold"]:.2f}\n')

        f.write(
            "#######################Overall Results "
            "(averaged across all scenarios):\n"
        )
        f.write(f"Number of scenarios: {len(results)}\n")
        f.write(f"Overall UNet MAE [deg]: {overall_unet_mae:.2f} deg\n")
        f.write(f"Overall Filter MAE [deg]: {overall_filter_mae:.2f} deg\n")
        f.write(f"Overall MSE [deg^2]: {overall_mse:.2f}\n")
        f.write(f"Overall CP Coverage: {overall_coverage:.3f}\n")

        f.write("#######################Per-scenario results:\n")
        for r in results:
            f.write(
                f"  {r['scenario']}: "
                f"UNet MAE={r['unet_mae']:.2f}, "
                f"Filter MAE={r['filter_mae']:.2f}, "
                f"MSE={r['doa_mse']:.2f}, "
                f"Coverage={r['cp_coverage']:.3f}\n"
            )
        f.write("#######################\n")

    # Log summary
    logging.info("\n" + "=" * 50)
    logging.info("All scenarios completed!")
    logging.info(f"Overall UNet MAE: {overall_unet_mae:.3f} deg")
    logging.info(f"Overall Filter MAE: {overall_filter_mae:.3f} deg")
    logging.info(f"Overall MSE: {overall_mse:.3f}")
    logging.info(f"Overall CP Coverage: {overall_coverage:.3f}")
    logging.info(f"Results CSV: {csv_file}")
    logging.info(f"Summary saved to: {summary_file}")
    logging.info("=" * 50)


def write_single_scenario_results(
    result: Dict[str, Any],
    args: Any,
    selected_checkpoint: Optional[str],
) -> None:
    """
    Write logging info and results.txt for single-scenario testing.

    Args:
        result:
            Dictionary containing test results for a single scenario.
        args:
            Argument namespace containing at least 'test_data_path'.
        selected_checkpoint:
            Path to the loaded checkpoint file, or None.
    """
    path = Path(args.test_data_path)
    subfolder = path.parent.name
    plots_dir = Path("plots") / subfolder
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_file = plots_dir / "results.txt"

    # Log results
    logging.info("\n" + "=" * 50)
    logging.info("Single scenario testing completed!")
    logging.info(f"UNet MAE: {result['unet_mae']:.3f} deg")
    logging.info(f"Filter MAE: {result['filter_mae']:.3f} deg")
    logging.info(f"MSE: {result['doa_mse']:.3f}")
    logging.info(f"CP Coverage: {result['cp_coverage']:.3f}")
    logging.info(f"Summary saved to: {summary_file}")
    logging.info("=" * 50)

    # Write results to file
    with summary_file.open("w") as f:
        f.write("#######################Configuration:\n")
        f.write(f"Data path: {args.test_data_path}\n")
        f.write(f'Checkpoints dir: {result["checkpoints_dir"]}\n')
        if selected_checkpoint:
            f.write(f"Checkpoint file: {selected_checkpoint}\n")
        f.write(f'RTF estimation method: {result["rtf_esti_method"]}\n')
        f.write(f'Feature operation: {result["feature_op"]}\n')
        f.write(f'Number of classes: {result["num_classes"]}\n')
        f.write(
            f'CP penalty for spacing (lambda): '
            f'{result["lambda_spacing"]:.2f}\n'
        )
        f.write(
            f'CP penalty for size (lambda): '
            f'{result["lambda_size"]:.2f}\n'
        )
        f.write(
            f'CP penalty for span (lambda): '
            f'{result["lambda_span"]:.2f}\n'
        )
        f.write(f'CP threshold: {result["threshold"]:.2f}\n')

        f.write("#######################Single Scenario Results:\n")
        f.write(f"Scenario: {result['scenario']}\n")
        f.write(f"Number of samples: {result['num_samples']}\n")
        f.write(f"UNet MAE [deg]: {result['unet_mae']:.2f} deg\n")
        f.write(f"Filter MAE [deg]: {result['filter_mae']:.2f} deg\n")
        f.write(f"MSE [deg^2]: {result['doa_mse']:.2f}\n")
        f.write(f"CP Coverage: {result['cp_coverage']:.3f}\n")
        f.write("#######################\n")


def get_h5_groups(h5_file_path: str) -> List[str]:
    """
    Get all top-level group names from an HDF5 file.

    Args:
        h5_file_path:
            Path to the HDF5 file.

    Returns:
        List of group names (strings).
    """
    groups: List[str] = []
    try:
        with h5py.File(h5_file_path, "r") as h5f:
            groups = list(h5f.keys())
        logging.info(f"Found {len(groups)} groups in {h5_file_path}")
    except Exception as e:
        logging.error(f"Error reading HDF5 file {h5_file_path}: {e}")

    return groups


def export_results_to_csv(
    results_list: List[Dict[str, Any]],
    test_data_path: str,
    output_file: str = "test_results.csv",
) -> Optional[str]:
    """
    Export a list of per-scenario results to a timestamped CSV file.

    Args:
        results_list:
            List of dictionaries containing test results. Expected keys:
                'scenario', 'unet_mae', 'filter_mae', 'doa_mse', 'cp_coverage'.
        test_data_path:
            Path to the test dataset (used to determine subfolder).
        output_file:
            Base name for the output CSV file (ignored for path, kept for API
            compatibility; a timestamped file is always created).

    Returns:
        The full path to the generated CSV file, or None if no results.
    """
    if not results_list:
        logging.warning("No results to export.")
        return None

    # Create plots directory structure consistent with other outputs
    path = Path(test_data_path)
    subfolder = path.parent.name
    plots_dir = Path("plots") / subfolder
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = plots_dir / f"test_results_{timestamp}.csv"

    with csv_path.open("w", newline="") as csvfile:
        fieldnames = ["scenario", "unet_mae", "filter_mae", "doa_mse", "cp_coverage"]
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        for result in results_list:
            writer.writerow(result)

    logging.info(f"Results exported to {csv_path}")
    return str(csv_path)


def write_result_to_txt(
    result: Dict[str, Any],
    args: Any,
    filepath: str = "result.txt",
) -> None:
    """
    Write a short summary for a single scenario (older/simple format).

    Args:
        result:
            Dictionary with keys like 'model_MAE' and 'filter_MAE'.
        args:
            Argument namespace containing at least:
                'test_data_path', 'lam_sp', 'lam_si', 'lam_span'.
        filepath:
            Output text file path.
    """
    with open(filepath, "w") as f:
        f.write("#######################\n")
        f.write("scenario parameters:\n")
        f.write(f"data path {args.test_data_path}\n")
        f.write(f"CP penalty for jumpy (lambda) {args.lam_sp:.2f} deg\n")
        f.write(f"CP penalty for size (lambda) {args.lam_si:.2f} deg\n")
        f.write(f"CP penalty for span (lambda) {args.lam_span:.2f} deg\n")
        f.write("#######################\n")
        f.write("Results:\n")
        f.write(f"unet MAE [deg]: {result['model_MAE']:.2f} deg\n")
        f.write(f"filter MAE [deg]: {result['filter_MAE']:.2f} deg\n")
        f.write("#######################\n")
