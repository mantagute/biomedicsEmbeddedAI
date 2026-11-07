"""
domain/signal/windowing.py
--------------------------
Estágio 2 do Pipeline DSP (Padrão Pipes & Filters): Sincronismo Multitaxa e
Janelamento Causal Hand-Made Multi-DoF.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from emg_loso.config import EMG_SAMPLE_RATE_HZ, KINEMATICS_SAMPLE_RATE_HZ
from emg_loso.domain.entities import SignalProcessingConfig
from emg_loso.domain.exceptions import InvalidSignalShapeError
from emg_loso.domain.signal.filtering import apply_highpass_filter


def compute_causal_window_bounds(
    n_emg_samples: int,
    n_angle_frames: int,
    window_samples: int,
    trim_seconds: float = 5.0,
    emg_fs_hz: float = EMG_SAMPLE_RATE_HZ,
    kinematics_fs_hz: float = KINEMATICS_SAMPLE_RATE_HZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula à mão (hand-made) os índices exatos de sincronismo EMG <-> Vicon."""
    first_frame = int(np.ceil(trim_seconds * kinematics_fs_hz))
    total_duration_s = n_angle_frames / kinematics_fs_hz
    last_frame = int(np.floor((total_duration_s - trim_seconds) * kinematics_fs_hz))

    if last_frame <= first_frame:
        raise InvalidSignalShapeError(
            f"Intervalo inválido após trim de {trim_seconds}s "
            f"(first_frame={first_frame}, last_frame={last_frame})."
        )

    frame_indices = np.arange(first_frame, last_frame, dtype=np.int32)
    label_times_s = frame_indices.astype(np.float64) / kinematics_fs_hz

    emg_end_idx = np.rint(label_times_s * emg_fs_hz).astype(np.int32)
    emg_start_idx = emg_end_idx - int(window_samples)

    valid_mask = (emg_start_idx >= 0) & (emg_end_idx <= n_emg_samples)

    return (
        emg_start_idx[valid_mask],
        emg_end_idx[valid_mask],
        frame_indices[valid_mask],
    )


def slice_windows_from_stream(
    emg_stream: np.ndarray,
    emg_start_idx: np.ndarray,
    window_samples: int,
) -> np.ndarray:
    """Fatia janelas 3D `(N, W, 128)` a partir de um stream contínuo via Broadcasting 2D."""
    offsets = np.arange(window_samples, dtype=np.int32)[None, :]
    sample_idx_grid = emg_start_idx[:, None] + offsets
    return emg_stream[sample_idx_grid]


def make_causal_windows_multidof(
    emg_array: np.ndarray,
    angles_df: pd.DataFrame,
    cfg: SignalProcessingConfig = SignalProcessingConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encadeia o Filter 1 (`filtering.py`) e o Filter 2 (`windowing.py`) para 1 gravação."""
    missing_cols = [c for c in cfg.target_dofs if c not in angles_df.columns]
    if missing_cols:
        raise InvalidSignalShapeError(
            f"Colunas cinemáticas ausentes: {missing_cols}. "
            f"Disponíveis: {list(angles_df.columns)}"
        )

    if cfg.highpass_hz is not None and cfg.highpass_hz > 0.0:
        emg_clean = apply_highpass_filter(
            emg_array,
            cutoff_hz=cfg.highpass_hz,
            order=cfg.filter_order,
            zero_phase=cfg.zero_phase,
        )
    else:
        emg_clean = emg_array

    emg_start_idx, _, valid_frames = compute_causal_window_bounds(
        n_emg_samples=len(emg_clean),
        n_angle_frames=len(angles_df),
        window_samples=cfg.window_samples,
        trim_seconds=cfg.trim_seconds,
    )

    windows = slice_windows_from_stream(
        emg_stream=emg_clean,
        emg_start_idx=emg_start_idx,
        window_samples=cfg.window_samples,
    )

    target_matrix = angles_df.loc[:, list(cfg.target_dofs)].to_numpy(dtype=np.float32)
    y_angles = target_matrix[valid_frames]

    return windows, y_angles, valid_frames
