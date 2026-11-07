"""
domain/signal/features.py
-------------------------
Estágio 3 do Pipeline DSP (Padrão Pipes & Filters): Extração de Descritores no
Domínio do Tempo (Time-Domain Features — TD) escrita Hand-Made em NumPy.
"""
from __future__ import annotations

from typing import Final
import numpy as np

from emg_loso.domain.exceptions import InvalidSignalShapeError

FEATURE_NAMES: Final[tuple[str, ...]] = ("rms", "mav", "wl")
N_FEATURES_PER_CHANNEL: Final[int] = len(FEATURE_NAMES)  # 3


def compute_rms(windows: np.ndarray) -> np.ndarray:
    """Calcula o Root Mean Square (RMS) ao longo do tempo (axis=1)."""
    return np.sqrt(np.mean(np.square(windows), axis=1, dtype=np.float32))


def compute_mav(windows: np.ndarray) -> np.ndarray:
    """Calcula o Mean Absolute Value (MAV) ao longo do tempo (axis=1)."""
    return np.mean(np.abs(windows), axis=1, dtype=np.float32)


def compute_wl(windows: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Calcula o Waveform Length (WL) ao longo do tempo (axis=1).

    Com `normalize=True` (padrão hardware-aware / AAC), divide por `W - 1` para
    manter a mesma ordem de grandeza de RMS/MAV em ponto fixo (`ap_fixed`).
    """
    abs_diffs = np.abs(np.diff(windows, axis=1))
    if normalize:
        return np.mean(abs_diffs, axis=1, dtype=np.float32)
    return np.sum(abs_diffs, axis=1, dtype=np.float32)


def extract_time_domain_features(
    windows: np.ndarray,
    normalize_wl: bool = True,
) -> np.ndarray:
    """Extrai e concatena `[RMS | MAV | WL]` formando a matriz plana `(N, 384)`."""
    if windows.ndim != 3:
        raise InvalidSignalShapeError(
            f"Esperado tensor 3D (n_windows, window_samples, n_channels), "
            f"mas recebido shape {windows.shape}."
        )

    rms = compute_rms(windows)
    mav = compute_mav(windows)
    wl = compute_wl(windows, normalize=normalize_wl)

    return np.concatenate([rms, mav, wl], axis=1).astype(np.float32, copy=False)
