"""
signal/filtering.py
-------------------
Estágio 1 do Pipeline DSP (Padrão Pipes & Filters): Condicionamento de Sinal.

Responsabilidade Única:
Projetar os coeficientes IIR Biquad (Seções de Segunda Ordem — SOS) do filtro
Butterworth passa-alta (20 Hz) e aplicá-los aos 128 canais de HD-sEMG tanto no
modo de referência de software (`zero_phase=True`) quanto no modo causal estrito
de hardware (`zero_phase=False`).

Isomorfismo com Hardware (FPGA Zynq):
A matriz `(2, 6)` retornada por `design_highpass_sos()` contém exatamente os
12 coeficientes constantes de duas seções Biquad em cascata, que mapeiam 1-para-1
para blocos DSP48E1 operando em AXI4-Stream.
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
    """Projeta a matriz de coeficientes SOS (Cascata de Biquads) do filtro passa-alta.

    Por que usar `scipy.signal.butter(..., output='sos')`?
    Quando a razão `cutoff_hz / fs_hz` é muito pequena (20 / 2052.52 ≈ 0.0097),
    os polos do filtro ficam extremamente próximos de z = 1 no círculo unitário.
    A representação polinomial direta `(b, a)` sofre perda severa de precisão
    numérica, enquanto a fatoração em Seções de Segunda Ordem (`sos`) preserva
    a estabilidade numérica e corresponde diretamente à topologia Biquad em hardware.

    Cada linha da matriz retornada `(n_sections, 6)` contém:
        `[b0, b1, b2, 1.0, a1, a2]`
    governando a equação de diferenças causal:
        `y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]`
    """
    return butter(N=order, Wn=cutoff_hz, btype="highpass", fs=fs_hz, output="sos")


def apply_highpass_filter(
    emg_array: np.ndarray,
    cutoff_hz: float = 20.0,
    order: int = 4,
    fs_hz: float = EMG_SAMPLE_RATE_HZ,
    zero_phase: bool = True,
) -> np.ndarray:
    """Aplica o filtro Butterworth passa-alta nos 128 canais de HD-sEMG (axis=0).

    Parâmetros
    ----------
    emg_array : np.ndarray, shape (n_samples, 128)
        Sinal contínuo bruto de HD-sEMG.
    cutoff_hz : float
        Frequência de corte passa-alta em Hz (padrão do artigo: 20.0 Hz).
    order : int
        Ordem do filtro Butterworth (padrão: 4).
    fs_hz : float
        Taxa de amostragem do amplificador HD-sEMG (2052.52 Hz).
    zero_phase : bool
        - `True` (Referencial de Software Offline): usa `sosfiltfilt` (varredura
          dupla frente-e-trás, zero distorção de fase / atraso de grupo nulo).
        - `False` (Modo Causal / Target FPGA): usa `sosfilt` (varredura única
          para frente, estritamente causal para operação em tempo real).

    Retorna
    -------
    np.ndarray, shape (n_samples, 128), dtype float32
        Sinal de HD-sEMG condicionado sem artefatos de movimento (< 20 Hz).
    """
    sos = design_highpass_sos(cutoff_hz=cutoff_hz, order=order, fs_hz=fs_hz)

    if zero_phase:
        filtered = sosfiltfilt(sos, emg_array, axis=0)
    else:
        filtered = sosfilt(sos, emg_array, axis=0)

    return filtered.astype(np.float32, copy=False)
