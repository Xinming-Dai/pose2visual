"""Shared utilities for neural encoding: CLI args, seeding, logging setup, co-bps metrics."""

import argparse
import logging
import os
import random
import warnings

import numpy as np
import torch
from scipy.special import gammaln

logger = logging.getLogger(__name__)

LATENT_KINDS = ('dynamic_vars',)


def setup_logging() -> None:
    """Configure root logging at INFO level; a no-op when handlers already exist.

    Ray Tune runs each trial in its own worker process, so trial functions call this too.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def get_encoding_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the neural encoding entry point.

    Args:
        argv: argument list to parse, e.g. `sys.argv[1:]`. If `None`, `argparse` falls
            back to reading `sys.argv` itself.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description='Neural Encoding')
    parser.add_argument(
        '--eid', type=str, required=True, help='Session / animal id (subfolder name)',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--neural_input_dir', type=str, default='input', help='Input directory for neural data',
    )
    parser.add_argument(
        '--latent_input_dir',
        type=str,
        default='input',
        help='Input directory for latents',
    )
    parser.add_argument(
        '--latent_kind',
        type=str,
        choices=LATENT_KINDS,
        default='dynamic_vars',
        help='Latent layout under --latent_input_dir: '
        'dynamic_vars → dynamic_vars_z/<eid>/dynamic_vars_z_trials.npz',
    )
    parser.add_argument(
        '--eval_task',
        type=str,
        choices=('encoding',),
        default='encoding',
        help='Evaluation task; only encoding is supported',
    )
    parser.add_argument(
        '--result_name',
        type=str,
        default=None,
        help='Basename for the saved .npy under the latent session dir (extension optional; '
        '.npy is added by numpy.save). Default: encoding_results_<latent_kind>.',
    )
    parser.add_argument(
        '--tune_storage_path',
        type=str,
        default=None,
        help='Ray Tune experiment root; trial logs go under this directory. '
        'Default: <latent_input_dir>/<layout subdir>, e.g. dynamic_vars_z.',
    )
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    """Seed all relevant RNGs for reproducibility.

    Args:
        seed: seed value applied to Python, numpy, and torch RNGs.
    """
    # PYTHONHASHSEED must be set before the interpreter starts to fully take effect, but
    # setting it here still helps downstream subprocesses that inherit the environment.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f'seed set to {seed}')


def neg_log_likelihood(
    rates: np.ndarray,
    spikes: np.ndarray,
    zero_warning: bool = True,
    threshold: float = 1e-9,
) -> float:
    """Calculate the Poisson negative log likelihood given rates and spikes.

    formula: -log(e^(-r) / n! * r^n) = r - n*log(r) + log(n!)

    Args:
        rates: numpy array containing rate predictions.
        spikes: numpy array containing true spike counts, same shape as `rates`.
        zero_warning: whether to warn about zero rate predictions.
        threshold: value used to replace zero rate predictions before taking `log`.

    Returns:
        Total negative log-likelihood of the data.
    """
    assert spikes.shape == rates.shape, (
        f'neg_log_likelihood: Rates and spikes should be of the same shape. '
        f'spikes: {spikes.shape}, rates: {rates.shape}'
    )

    if np.any(np.isnan(spikes)):
        mask = np.isnan(spikes)
        rates = rates[~mask]
        spikes = spikes[~mask]

    assert not np.any(np.isnan(rates)), 'neg_log_likelihood: NaN rate predictions found'

    assert np.all(rates >= 0), 'neg_log_likelihood: Negative rate predictions found'
    if np.any(rates == 0):
        if zero_warning:
            warnings.warn(
                'neg_log_likelihood: Zero rate predictions found. Replacing zeros with 1e-9',
                stacklevel=2,
            )
        rates[rates == 0] = threshold
    result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
    return np.sum(result)


def bits_per_spike(rates: np.ndarray, spikes: np.ndarray, threshold: float = 1e-9) -> float:
    """Compute bits per spike of rate predictions given spikes.

    Bits per spike is the difference between the log-likelihoods (in base 2) of the rate
    predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Args:
        rates: 3d numpy array containing rate predictions.
        spikes: 3d numpy array containing true spike counts.
        threshold: value used to replace zero rate predictions.

    Returns:
        Bits per spike of rate predictions.
    """
    nll_model = neg_log_likelihood(rates, spikes, threshold=threshold)
    null_rates = np.tile(
        np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
        spikes.shape[:-1] + (1,),
    )
    nll_null = neg_log_likelihood(null_rates, spikes, threshold=threshold)
    return (nll_null - nll_model) / np.nansum(spikes) / np.log(2)


def _std(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score `arr` over its leading two axes (trial, time), keeping the last axis.

    Args:
        arr: array of shape `[K, T, N]`.

    Returns:
        A tuple `(normalized_arr, mean, std)` where `mean` and `std` have shape `[1, N]`.
    """
    mean = np.nanmean(arr, axis=(0, 1), keepdims=False).reshape(1, -1)
    std = np.nanstd(arr, axis=(0, 1), keepdims=False).reshape(1, -1)
    std = np.clip(std, 1e-8, None)
    arr = (arr - mean) / std
    return arr, mean, std
