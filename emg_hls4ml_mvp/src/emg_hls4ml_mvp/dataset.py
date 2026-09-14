from __future__ import annotations

from pathlib import Path

import numpy as np

from emg_hls4ml_mvp.data_loading import load_angles_csv, load_emg_csv, recording_paths
from emg_hls4ml_mvp.features import extract_time_domain_features
from emg_hls4ml_mvp.windowing import make_causal_windows


def build_feature_dataset(
    data_root: str | Path,
    subject: str,
    recording_id: str,
    target_column: str = "index_z",
    window_ms: float = 150.0,
    trim_seconds: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build feature matrix X and target vector y for one recording.

    This function is the first reproducible pipeline: CSV tables enter, numpy
    arrays leave. X has shape (n_windows, n_features), y has shape (n_windows,).
    """
    emg_path, angles_path = recording_paths(data_root, subject, recording_id)
    emg = load_emg_csv(emg_path)
    angles = load_angles_csv(angles_path)

    windows, y, label_indices = make_causal_windows(
        emg,
        angles,
        target_column=target_column,
        window_ms=window_ms,
        trim_seconds=trim_seconds,
    )
    X = extract_time_domain_features(windows)

    return X, y, label_indices
