"""
features.py
-----------
Extração de features de domínio do tempo a partir de janelas de EMG.

Cada feature reduz o eixo temporal (axis=1) de um tensor de janelas,
produzindo um valor escalar por canal por janela.

Features implementadas:
    RMS (Root Mean Square)  — energia do sinal; sensível à amplitude.
    MAV (Mean Absolute Value) — versão mais robusta a outliers que o RMS.
    WL  (Waveform Length)   — comprimento de onda; captura complexidade e frequência.

Referência: Phinyomark et al. (2012), "Feature reduction and selection for
EMG signal classification."

Formato de entrada esperado:
    windows : np.ndarray, shape (n_windows, n_samples, n_channels)

Formato de saída:
    features : np.ndarray, shape (n_windows, n_channels * 3)
    Ordem da concatenação: [RMS | MAV | WL]
"""
from __future__ import annotations

import numpy as np


def compute_rms(windows: np.ndarray) -> np.ndarray:
    """Root Mean Square por canal.

    Mede a energia média do sinal. Valores altos indicam contração muscular intensa.
    Shape: (n_windows, n_channels).
    """
    return np.sqrt(np.mean(np.square(windows), axis=1))


def compute_mav(windows: np.ndarray) -> np.ndarray:
    """Mean Absolute Value por canal.

    Equivalente ao RMS mas sem elevar ao quadrado; menos sensível a picos isolados.
    Shape: (n_windows, n_channels).
    """
    return np.mean(np.abs(windows), axis=1)


def compute_wl(windows: np.ndarray) -> np.ndarray:
    """Waveform Length por canal.

    Soma das diferenças absolutas consecutivas ao longo do tempo.
    Captura a complexidade e oscilação de frequência do sinal.
    Shape: (n_windows, n_channels).
    """
    # np.diff(axis=1) calcula a diferença entre amostras consecutivas no tempo.
    return np.sum(np.abs(np.diff(windows, axis=1)), axis=1)


def extract_time_domain_features(windows: np.ndarray) -> np.ndarray:
    """Extrai RMS, MAV e WL e concatena em um vetor de features por janela.

    Entrada : (n_windows, n_samples, n_channels)
    Saída   : (n_windows, n_channels * 3)   — ordem: [RMS | MAV | WL]
    """
    rms = compute_rms(windows)
    mav = compute_mav(windows)
    wl  = compute_wl(windows)

    # Concatenar ao longo do eixo de features (axis=1).
    return np.concatenate([rms, mav, wl], axis=1).astype(np.float32)
