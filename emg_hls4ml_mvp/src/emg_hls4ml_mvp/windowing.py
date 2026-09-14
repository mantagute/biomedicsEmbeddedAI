"""
windowing.py
------------
Criação de janelas causais de EMG alinhadas à cinemática.

Problema de alinhamento temporal:
    EMG é amostrado a 2052.52 Hz; cinemática (Vicon) a 100 Hz.
    Para cada instante de label cinemático (t_label), a janela de EMG
    é o segmento imediatamente anterior de duração `window_ms`.

    Diagrama:
        EMG:  |---------------------------[  window_samples  ]--|
                                                                ^ t_label (100 Hz)
        Label:                                                  y = ângulo do dedo

    Isso é "causal" porque a janela usa apenas amostras passadas, sem lookahead.
    Num sistema embarcado real, isso significa que o FPGA pode computar a
    inferência assim que a janela chegar, sem esperar amostras futuras.
"""
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
    """Cria janelas causais de EMG alinhadas a um alvo cinemático contínuo.

    Parâmetros
    ----------
    emg : pd.DataFrame
        Sinal EMG, shape (n_emg_samples, n_channels). Colunas ch_001 ... ch_128.
    angles : pd.DataFrame
        Ângulos cinemáticos, shape (n_label_frames, n_dofs).
    target_column : str
        Nome da coluna de ângulo usada como label (ex: "index_z").
    window_ms : float
        Duração da janela causal de EMG em milissegundos.
    trim_seconds : float
        Segundos descartados no início e no fim do sinal para remover
        transientes de movimento (sujeito ainda acelerando/desacelerando).

    Retorna
    -------
    windows : np.ndarray, shape (n_windows, window_samples, n_channels)
        Tensor de janelas de EMG.
    labels : np.ndarray, shape (n_windows,)
        Valor do ângulo alvo no instante de cada janela.
    label_indices : np.ndarray, shape (n_windows,)
        Índice do frame cinemático correspondente a cada janela.
        Útil para reconstruir a série temporal e plotar.
    """
    if target_column not in angles.columns:
        raise KeyError(
            f"Coluna '{target_column}' não encontrada. "
            f"Disponíveis: {list(angles.columns)}"
        )

    # --- Conversão para NumPy ---
    # Pandas só foi necessário até aqui para leitura e validação das colunas.
    # A partir deste ponto, toda operação numérica usa NumPy.
    emg_array    = emg.to_numpy(dtype=np.float32)             # (n_emg_samples, 128)
    target_array = angles[target_column].to_numpy(dtype=np.float32)  # (n_label_frames,)

    # --- Calcular tamanho da janela em amostras de EMG ---
    # 150 ms × 2052.52 Hz ≈ 308 amostras por janela.
    window_samples = int(round(EMG_SAMPLE_RATE_HZ * window_ms / 1000.0))

    # --- Definir intervalo válido de frames cinemáticos ---
    # trim_seconds remove bordas ruidosas; a conversão garante índices inteiros.
    first_valid_frame = int(np.ceil(trim_seconds * KINEMATICS_SAMPLE_RATE_HZ))
    last_valid_frame  = int(np.floor(
        (len(angles) / KINEMATICS_SAMPLE_RATE_HZ - trim_seconds)
        * KINEMATICS_SAMPLE_RATE_HZ
    ))

    # --- Montar janelas de forma vetorizada ---
    # Para cada frame cinemático válido, calculamos o índice final da janela
    # de EMG e extraímos o slice correspondente via broadcasting de índices.
    frame_indices = np.arange(first_valid_frame, last_valid_frame)

    # Instante de tempo (segundos) de cada frame cinemático.
    label_times_s = frame_indices / KINEMATICS_SAMPLE_RATE_HZ

    # Índice de fim de cada janela no array de EMG (amostra mais recente).
    emg_end_indices   = np.round(label_times_s * EMG_SAMPLE_RATE_HZ).astype(np.int32)
    emg_start_indices = emg_end_indices - window_samples

    # Descartar frames em que a janela ultrapassa as bordas do array de EMG.
    valid_mask = (emg_start_indices >= 0) & (emg_end_indices <= len(emg_array))
    emg_end_indices   = emg_end_indices[valid_mask]
    emg_start_indices = emg_start_indices[valid_mask]
    valid_frames      = frame_indices[valid_mask]

    # Construir tensor de janelas: shape (n_windows, window_samples, n_channels).
    # np.stack converte a lista de slices 2D em um tensor 3D.
    windows = np.stack([
        emg_array[start:end]
        for start, end in zip(emg_start_indices, emg_end_indices)
    ]).astype(np.float32)

    labels        = target_array[valid_frames].astype(np.float32)
    label_indices = valid_frames.astype(np.int32)

    return windows, labels, label_indices
