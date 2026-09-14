from __future__ import annotations

import numpy as np
import pandas as pd

from emg_hls4ml_mvp.data_loading import EMG_SAMPLE_RATE_HZ, KINEMATICS_SAMPLE_RATE_HZ


def make_causal_windows(
    emg: pd.DataFrame,
    angles: pd.DataFrame,
    target_column: str,
    window_ms: float = 150.0,
    trim_seconds: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create causal EMG windows aligned to one kinematic target column.

    Inputs are pandas tables for readability at the dataset boundary. Inside the
    function they become numpy arrays, because window slicing and ML tensors are
    simpler as numeric matrices. Output shape is (windows, samples, channels).
    """
    if target_column not in angles.columns:
        raise KeyError(f"Unknown target column: {target_column}")

    # DataFrame -> ndarray: from named table to numeric matrix.
    emg_values = emg.to_numpy(dtype=np.float32)
    target_values = angles[target_column].to_numpy(dtype=np.float32)

    # 150 ms at 2052.52 Hz becomes about 308 EMG samples.
    window_samples = int(round(EMG_SAMPLE_RATE_HZ * window_ms / 1000.0))
    first_label_index = int(np.ceil(trim_seconds * KINEMATICS_SAMPLE_RATE_HZ))
    last_label_index = int(np.floor((len(angles) / KINEMATICS_SAMPLE_RATE_HZ - trim_seconds) * KINEMATICS_SAMPLE_RATE_HZ))

    # Lists are convenient while the number of valid windows is being built.
    windows = []
    labels = []
    label_indices = []

    for label_index in range(first_label_index, last_label_index):
        label_time_seconds = label_index / KINEMATICS_SAMPLE_RATE_HZ
        emg_end = int(round(label_time_seconds * EMG_SAMPLE_RATE_HZ))
        emg_start = emg_end - window_samples

        if emg_start < 0 or emg_end > len(emg_values):
            continue

        windows.append(emg_values[emg_start:emg_end])
        labels.append(target_values[label_index])
        label_indices.append(label_index)

    return (
        np.stack(windows).astype(np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(label_indices, dtype=np.int32),
    )
