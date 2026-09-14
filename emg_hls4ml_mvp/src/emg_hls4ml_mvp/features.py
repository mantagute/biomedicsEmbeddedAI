from __future__ import annotations

import numpy as np


def extract_time_domain_features(windows: np.ndarray) -> np.ndarray:
    """Extract RMS, MAV, and WL features from EMG windows.

    Expected input shape: (n_windows, n_samples, n_channels). The feature
    operations reduce axis 1, the time axis, keeping one value per channel.
    Output shape: (n_windows, n_channels * 3).
    """
    # RMS, MAV, and WL each produce one value per channel for every window.
    rms = np.sqrt(np.mean(np.square(windows), axis=1))
    mav = np.mean(np.abs(windows), axis=1)
    wl = np.sum(np.abs(np.diff(windows, axis=1)), axis=1)

    return np.concatenate([rms, mav, wl], axis=1).astype(np.float32)
