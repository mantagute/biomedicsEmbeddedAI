"""
signal/features.py
------------------
Estágio 3 do Pipeline DSP (Padrão Pipes & Filters): Extração de Descritores no
Domínio do Tempo (Time-Domain Features — TD) escrita Hand-Made em NumPy.

Responsabilidade Única:
Reduzir o eixo temporal `axis=1` do tensor de janelas `(N, W, 128)` para três
descritores físicos por canal:
1. RMS (Root Mean Square) — Potência efetiva / recrutamento de unidades motoras.
2. MAV (Mean Absolute Value) — Envelope retificado médio (robusto a picos isolados).
3. WL  (Waveform Length / AAC) — Complexidade temporal e taxa de oscilação (frequência).

Por que Hand-Made em vez de bibliotecas como `tsfresh` ou `scipy.stats`?
1. Performance e Controle de Memória: Reduções matriciais diretas no `axis=1` em
   `float32` processam 3.500 janelas de 128 canais em ~15 ms sem criar objetos
   intermediários.
2. Co-Design Hardware/Software (Prevenção de Overflow em Ponto Fixo na FPGA):
   O cálculo clássico de WL soma `|x[t] - x[t-1]|` ao longo de `W-1` amostras
   sem dividir por `W-1`. Para `W = 308`, isso multiplica a escala numérica por ~307x
   em relação ao RMS e MAV, estourando os bits inteiros de um `ap_fixed<16,6>` no
   Vitis HLS. Nossa implementação inclui `normalize_wl=True` (equivalente ao
   Average Amplitude Change — AAC de Phinyomark et al., 2012), mantendo RMS, MAV
   e WL na mesma ordem de grandeza física para síntese em ponto fixo.
"""
from __future__ import annotations

from typing import Final
import numpy as np

FEATURE_NAMES: Final[tuple[str, ...]] = ("rms", "mav", "wl")
N_FEATURES_PER_CHANNEL: Final[int] = len(FEATURE_NAMES)  # 3


def compute_rms(windows: np.ndarray) -> np.ndarray:
    """Calcula o Root Mean Square (RMS) ao longo do tempo (axis=1).

    Equação:
        RMS = sqrt( (1 / W) * sum_{t=0}^{W-1} x[t]^2 )

    Entrada : (n_windows, window_samples, n_channels)
    Saída   : (n_windows, n_channels) em float32
    """
    return np.sqrt(np.mean(np.square(windows), axis=1, dtype=np.float32))


def compute_mav(windows: np.ndarray) -> np.ndarray:
    """Calcula o Mean Absolute Value (MAV) ao longo do tempo (axis=1).

    Equação:
        MAV = (1 / W) * sum_{t=0}^{W-1} |x[t]|

    Entrada : (n_windows, window_samples, n_channels)
    Saída   : (n_windows, n_channels) em float32
    """
    return np.mean(np.abs(windows), axis=1, dtype=np.float32)


def compute_wl(windows: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Calcula o Waveform Length (WL) ao longo do tempo (axis=1).

    Equação:
        - Se `normalize=True` (Padrão Hardware-Aware / AAC):
              WL_norm = (1 / (W - 1)) * sum_{t=1}^{W-1} |x[t] - x[t-1]|
          Mantém a feature invariante ao tamanho da janela `W` e na mesma escala
          dinâmica de RMS/MAV (evitando saturação de bits inteiros em `ap_fixed`).
        - Se `normalize=False` (Soma bruta clássica):
              WL_raw  = sum_{t=1}^{W-1} |x[t] - x[t-1]|

    Nota sobre o StandardScaler:
        Como `WL_norm = WL_raw / (W - 1)` é um escalonamento linear por constante
        positiva, após a padronização Z-score (`StandardScaler`) em ponto flutuante
        os dois produzem tensores 100% idênticos! A vantagem do `normalize=True`
        aparece quando exportamos os acumuladores para ponto fixo na FPGA.

    Entrada : (n_windows, window_samples, n_channels)
    Saída   : (n_windows, n_channels) em float32
    """
    # Diferença de primeira ordem no tempo: x[:, 1:, :] - x[:, :-1, :]
    abs_diffs = np.abs(np.diff(windows, axis=1))
    if normalize:
        return np.mean(abs_diffs, axis=1, dtype=np.float32)
    return np.sum(abs_diffs, axis=1, dtype=np.float32)


def extract_time_domain_features(
    windows: np.ndarray,
    normalize_wl: bool = True,
) -> np.ndarray:
    """Extrai e concatena `[RMS | MAV | WL]` formando a matriz plana `(N, 384)`.

    Layout exato das 384 colunas de saída (essencial para o Filter 4 `topography.py`):
        - Colunas   0 .. 127 : RMS dos canais 1..128 (0..63 = EDC, 64..127 = FDS)
        - Colunas 128 .. 255 : MAV dos canais 1..128 (128..191 = EDC, 192..255 = FDS)
        - Colunas 256 .. 383 : WL  dos canais 1..128 (256..319 = EDC, 320..383 = FDS)

    Parâmetros
    ----------
    windows : np.ndarray, shape (n_windows, window_samples, 128)
        Tensor 3D de janelas causais condicionado pelo Filter 1 e Filter 2.
    normalize_wl : bool
        Se True (padrão), divide o WL por `W - 1` para preservar a escala de ponto fixo.

    Retorna
    -------
    X_flat : np.ndarray, shape (n_windows, 384), dtype float32
        Matriz de features 2D pronta para alimentar o MLP ou ser remodelada
        para `(n_windows, 8, 8, 6)` pelo `topography.py`.
    """
    if windows.ndim != 3:
        raise ValueError(
            f"Esperado tensor 3D (n_windows, window_samples, n_channels), "
            f"mas recebido shape {windows.shape}."
        )

    rms = compute_rms(windows)
    mav = compute_mav(windows)
    wl = compute_wl(windows, normalize=normalize_wl)

    return np.concatenate([rms, mav, wl], axis=1).astype(np.float32, copy=False)
