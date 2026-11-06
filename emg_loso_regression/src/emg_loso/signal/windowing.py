"""
signal/windowing.py
-------------------
Estágio 2 do Pipeline DSP (Padrão Pipes & Filters): Sincronismo Multitaxa e
Janelamento Causal Hand-Made Multi-DoF.

Responsabilidade Única:
Alinhar os dois relógios físicos do experimento (EMG a 2052.52 Hz e câmera Vicon
a 100.0 Hz) sem reamostragem destrutiva e extrair janelas causais de 150 ms
sincronizadas aos 5 Graus de Liberdade (DoFs) dos dedos.

Isomorfismo com Hardware (FPGA Zynq):
O cálculo explícito de ponteiros (`compute_causal_window_bounds`) e a matriz de
endereçamento 2D (`sample_idx_grid`) replicam exatamente a leitura de um buffer
circular (Ring Buffer em BRAM) disparada a cada 10 ms (100 Hz).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from emg_loso.config import (
    EMG_SAMPLE_RATE_HZ,
    KINEMATICS_SAMPLE_RATE_HZ,
    SignalProcessingConfig,
)
from emg_loso.signal.filtering import apply_highpass_filter


def compute_causal_window_bounds(
    n_emg_samples: int,
    n_angle_frames: int,
    window_samples: int,
    trim_seconds: float = 5.0,
    emg_fs_hz: float = EMG_SAMPLE_RATE_HZ,
    kinematics_fs_hz: float = KINEMATICS_SAMPLE_RATE_HZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula à mão (hand-made) os índices exatos de sincronismo EMG <-> Vicon.

    Por que escrever essa aritmética na mão em vez de usar `scipy.signal.resample`?
    Reamostrar o EMG de 2052.52 Hz para 2000 Hz introduz oscilações espúrias
    (ringing de Gibbs) e exige processamento extra no sistema embarcado. Mantendo
    as taxas nativas, mapeamos cada instante `t_k = k / 100.0` diretamente para
    a amostra mais recente de EMG disponível no buffer:
        `n_end(k)   = round(t_k * 2052.52)`
        `n_start(k) = n_end(k) - W`

    Retorna
    -------
    emg_start_idx : np.ndarray (int32), shape (n_valid_windows,)
        Índice inicial (inclusivo) de cada janela no array de EMG.
    emg_end_idx : np.ndarray (int32), shape (n_valid_windows,)
        Índice final (exclusivo) de cada janela no array de EMG.
    valid_frames : np.ndarray (int32), shape (n_valid_windows,)
        Índice do frame correspondente na tabela de ângulos do Vicon.
    """
    first_frame = int(np.ceil(trim_seconds * kinematics_fs_hz))
    total_duration_s = n_angle_frames / kinematics_fs_hz
    last_frame = int(np.floor((total_duration_s - trim_seconds) * kinematics_fs_hz))

    if last_frame <= first_frame:
        raise ValueError(
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
    """Fatia janelas 3D `(N, W, 128)` a partir de um stream contínuo via Broadcasting 2D.

    Separar o fatiamento puro permite testar e reutilizar a lógica de buffer
    circular independentemente do DataFrame de ângulos.
    """
    offsets = np.arange(window_samples, dtype=np.int32)[None, :]  # (1, W)
    sample_idx_grid = emg_start_idx[:, None] + offsets            # (N, W)
    return emg_stream[sample_idx_grid]                            # (N, W, 128)


def make_causal_windows_multidof(
    emg_array: np.ndarray,
    angles_df: pd.DataFrame,
    cfg: SignalProcessingConfig = SignalProcessingConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encadeia o Filter 1 (`filtering.py`) e o Filter 2 (`windowing.py`) para 1 gravação.

    Retorna
    -------
    windows : np.ndarray, shape (n_windows, window_samples, 128)
        Tensor 3D `float32` das janelas causais condicionadas.
    y_angles : np.ndarray, shape (n_windows, n_dofs)
        Matriz `float32` com os ângulos dos 5 dedos (`thumb_z` a `little_z`).
    frame_indices : np.ndarray, shape (n_windows,)
        Vetor `int32` com o índice do frame original a 100 Hz.
    """
    missing_cols = [c for c in cfg.target_dofs if c not in angles_df.columns]
    if missing_cols:
        raise KeyError(
            f"Colunas cinemáticas ausentes: {missing_cols}. "
            f"Disponíveis: {list(angles_df.columns)}"
        )

    # Filter 1: Condicionamento passa-alta na série contínua (antes de cortar janelas)
    if cfg.highpass_hz is not None and cfg.highpass_hz > 0.0:
        emg_clean = apply_highpass_filter(
            emg_array,
            cutoff_hz=cfg.highpass_hz,
            order=cfg.filter_order,
            zero_phase=cfg.zero_phase,
        )
    else:
        emg_clean = emg_array

    # Filter 2: Sincronismo e fatiamento causal vetorizado
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
