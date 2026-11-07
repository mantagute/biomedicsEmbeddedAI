"""
domain/signal/filtering.py
--------------------------
Estágio 1 do Pipeline DSP (Padrão Pipes & Filters): Condicionamento de Sinal.

Responsabilidade Única:
Projetar os coeficientes IIR Biquad (Seções de Segunda Ordem — SOS) do filtro
Butterworth passa-alta (20 Hz) e aplicá-los aos 128 canais de HD-sEMG tanto no
modo de referência de software (`zero_phase=True`) quanto no modo causal estrito
de hardware (`zero_phase=False`).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from emg_loso.config import EMG_SAMPLE_RATE_HZ


def design_highpass_sos(
    cutoff_hz: float = 20.0,
    order: int = 4,
    fs_hz: float = EMG_SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Projeta a matriz de coeficientes SOS (Cascata de Biquads) do filtro passa-alta."""
    return butter(N=order, Wn=cutoff_hz, btype="highpass", fs=fs_hz, output="sos")


def apply_highpass_filter(
    emg_array: np.ndarray,
    cutoff_hz: float = 20.0,
    order: int = 4,
    fs_hz: float = EMG_SAMPLE_RATE_HZ,
    zero_phase: bool = True,
) -> np.ndarray:
    """Aplica o filtro Butterworth passa-alta nos 128 canais de HD-sEMG (axis=0)."""
    sos = design_highpass_sos(cutoff_hz=cutoff_hz, order=order, fs_hz=fs_hz)

    if zero_phase:
        filtered = sosfiltfilt(sos, emg_array, axis=0)
    else:
        filtered = sosfilt(sos, emg_array, axis=0)

    return filtered.astype(np.float32, copy=False)
