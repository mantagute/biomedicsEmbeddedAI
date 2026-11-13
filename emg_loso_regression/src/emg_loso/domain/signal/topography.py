"""
domain/signal/topography.py
---------------------------
Estágio 4 do Pipeline DSP (Padrão Pipes & Filters): Mapeamento Espacial Topográfico 2D.

Responsabilidade Única:
Reconstruir a geometria física das duas matrizes de eletrodos HD-sEMG (8x8 no
músculo extensor EDC + 8x8 no músculo flexor FDS) a partir do vetor plano de
features `(N, 384)`, produzindo o tensor espacial `(N, 8, 8, 6)` em layout
channels-last (padrão TensorFlow/Keras e hls4ml).

Por que Hand-Made em NumPy puro?
Um `.reshape(N, 8, 8, 6)` direto sobre `(N, 384)` misturaria canais adjacentes de
RMS em eixos errados devido à ordenação row-major do C/NumPy. Fazer o
mapeamento explícito em dois passos (`reshape(N, 6, 8, 8)` -> `transpose(0, 2, 3, 1)`)
garante que cada coordenada `(r, c)` contenha exatamente os 6 descritores
alinhados daquela posição física nos dois músculos:
    canal 0: RMS_EDC(r, c)
    canal 1: RMS_FDS(r, c)
    canal 2: MAV_EDC(r, c)
    canal 3: MAV_FDS(r, c)
    canal 4: WL_EDC(r, c)
    canal 5: WL_FDS(r, c)
"""
from __future__ import annotations

from typing import Final
import numpy as np
import pandas as pd

from emg_loso.config import GRID_COLS, GRID_ROWS, N_EMG_CHANNELS, N_GRIDS
from emg_loso.domain.entities import SignalProcessingConfig
from emg_loso.domain.exceptions import InvalidSignalShapeError
from emg_loso.domain.signal.features import (
    N_FEATURES_PER_CHANNEL,
    extract_time_domain_features,
)
from emg_loso.domain.signal.windowing import make_causal_windows_multidof

N_TOPOGRAPHIC_CHANNELS: Final[int] = N_GRIDS * N_FEATURES_PER_CHANNEL  # 6
EXPECTED_FLAT_DIM: Final[int] = N_EMG_CHANNELS * N_FEATURES_PER_CHANNEL  # 384


def flat_to_topographic_grid(x_flat: np.ndarray) -> np.ndarray:
    """Converte matriz plana `(N, 384)` para tensor topográfico 2D `(N, 8, 8, 6)`.

    Parâmetros
    ----------
    x_flat : np.ndarray, shape (n_windows, 384)
        Matriz `[RMS | MAV | WL]` gerada por `extract_time_domain_features`.

    Retorna
    -------
    x_grid : np.ndarray, shape (n_windows, 8, 8, 6), dtype float32
        Tensor 4D channels-last pronto para convoluções 2D espaciais.
    """
    if x_flat.ndim != 2 or x_flat.shape[1] != EXPECTED_FLAT_DIM:
        raise InvalidSignalShapeError(
            f"Esperado shape (N, {EXPECTED_FLAT_DIM}) para conversão topográfica, "
            f"mas recebido {x_flat.shape}."
        )

    n_windows = x_flat.shape[0]
    # Passo 1: separa os 6 blocos contíguos de 64 eletrodos -> (N, 6, 8, 8)
    maps_first = x_flat.reshape(n_windows, N_TOPOGRAPHIC_CHANNELS, GRID_ROWS, GRID_COLS)
    # Passo 2: move o eixo dos 6 mapas para o final (channels-last) -> (N, 8, 8, 6)
    x_grid = np.ascontiguousarray(maps_first.transpose(0, 2, 3, 1), dtype=np.float32)
    return x_grid


def topographic_grid_to_flat(x_grid: np.ndarray) -> np.ndarray:
    """Operação inversa exata: converte `(N, 8, 8, 6)` de volta para `(N, 384)`.

    Útil para testes de bijeção (garantia de zero embaralhamento de canais) e
    para re-achatamento quando comparando arquiteturas.
    """
    expected_tail = (GRID_ROWS, GRID_COLS, N_TOPOGRAPHIC_CHANNELS)
    if x_grid.ndim != 4 or x_grid.shape[1:] != expected_tail:
        raise InvalidSignalShapeError(
            f"Esperado shape (N, {GRID_ROWS}, {GRID_COLS}, {N_TOPOGRAPHIC_CHANNELS}), "
            f"mas recebido {x_grid.shape}."
        )

    n_windows = x_grid.shape[0]
    maps_first = x_grid.transpose(0, 3, 1, 2)
    return np.ascontiguousarray(maps_first.reshape(n_windows, EXPECTED_FLAT_DIM), dtype=np.float32)


def extract_recording_representations(
    emg_array: np.ndarray,
    angles_df: pd.DataFrame,
    cfg: SignalProcessingConfig = SignalProcessingConfig(),
    normalize_wl: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Executa o pipeline Pipes & Filters completo (Filter 1 -> 2 -> 3 -> 4) em 1 gravação.

    Retorna
    -------
    x_flat : np.ndarray, shape (n_windows, 384), dtype float32
        Representação vetorial para o baseline Deep MLP.
    x_grid : np.ndarray, shape (n_windows, 8, 8, 6), dtype float32
        Representação topográfica 2D para a 2D-CNN.
    y_angles : np.ndarray, shape (n_windows, n_dofs), dtype float32
        Ângulos articulares simultâneos dos 5 dedos (`thumb_z` a `little_z`).
    frame_indices : np.ndarray, shape (n_windows,), dtype int32
        Índices originais dos frames cinemáticos (100 Hz).
    """
    windows, y_angles, frame_indices = make_causal_windows_multidof(
        emg_array=emg_array,
        angles_df=angles_df,
        cfg=cfg,
    )
    x_flat = extract_time_domain_features(windows, normalize_wl=normalize_wl)
    x_grid = flat_to_topographic_grid(x_flat)
    return x_flat, x_grid, y_angles, frame_indices
