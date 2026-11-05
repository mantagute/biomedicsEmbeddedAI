"""
windowing.py
------------
Condicionamento de sinal (filtragem passa-alta de 20 Hz) e janelamento causal
feito à mão (hand-crafted em NumPy puro) sincronizado aos 5 Graus de Liberdade
(DoFs) cinemáticos da mão.

Filosofia de Engenharia deste Módulo (Biblioteca vs. Hand-Made):
1. Onde usamos biblioteca padrão de mercado (`scipy.signal`):
   No projeto dos coeficientes do filtro Butterworth em Seções de Segunda Ordem
   (SOS) via transformação bilinear e na execução em C (`sosfilt` / `sosfiltfilt`).
   Reescrever um solver de raízes polinomiais de Butterworth ou o método de
   Gustafsson para condições de contorno em Python puro seria lento e propenso
   a erros de ponto flutuante.
2. Onde fazemos Hand-Made (Aritmética Matricial Explícita em NumPy):
   No sincronismo multitaxa (2052.52 Hz <-> 100 Hz) e no fatiamento das janelas
   causais 3D. Não usamos bibliotecas "caixa-preta" de séries temporais (como
   `tsfresh`, `mne` ou `tf.keras.utils.timeseries_dataset_from_array`) porque
   elas escondem o alinhamento temporal exato entre duas taxas de amostragem
   irracionais (2052.52 Hz e 100 Hz), dificultam a auditoria contra *data leakage*
   e não podem ser traduzidas diretamente para ponteiros de buffer circular em C++/HLS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfiltfilt

from emg_loso.config import (
    EMG_SAMPLE_RATE_HZ,
    KINEMATICS_SAMPLE_RATE_HZ,
    SignalProcessingConfig,
)


def design_highpass_sos(
    cutoff_hz: float = 20.0,
    order: int = 4,
    fs_hz: float = EMG_SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Projeta a matriz de coeficientes SOS (Cascata de Biquads) do filtro passa-alta.

    Por que exportar a matriz SOS separadamente?
    Cada linha da matriz retornada `(n_sections, 6)` contém exatamente os 6
    coeficientes `[b0, b1, b2, 1.0, a1, a2]` de um estágio IIR Biquad:
        y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
    Separar o cálculo dos coeficientes permite exportar esses exatos valores
    constantes para um header `.h` na FPGA/ARM caso o pré-processamento seja
    embarcado.
    """
    return butter(N=order, Wn=cutoff_hz, btype="highpass", fs=fs_hz, output="sos")


def apply_highpass_filter(
    emg_array: np.ndarray,
    cutoff_hz: float = 20.0,
    order: int = 4,
    fs_hz: float = EMG_SAMPLE_RATE_HZ,
    zero_phase: bool = True,
) -> np.ndarray:
    """Filtra os 128 canais de HD-sEMG ao longo do eixo temporal (axis=0).

    Suporta os dois cenários do projeto:
    - `zero_phase=True` (Referencial de Software Offline):
      Aplica o filtro para frente e para trás (`sosfiltfilt`). Distorção de fase
      nula (0 ms de atraso de grupo), preservando o alinhamento temporal exato
      dos picos de MUAPs com a câmera Vicon.
    - `zero_phase=False` (Modo Causal / Target FPGA):
      Aplica apenas a varredura causal para frente (`sosfilt`), idêntica à
      equação de diferenças executada em tempo real amostra por amostra.
    """
    sos = design_highpass_sos(cutoff_hz=cutoff_hz, order=order, fs_hz=fs_hz)

    if zero_phase:
        # ATENÇÃO: sosfiltfilt dobra a ordem efetiva do filtro (4ª -> 8ª ordem)
        # e exige a gravação completa em memória (modo offline).
        filtered = sosfiltfilt(sos, emg_array, axis=0)
    else:
        # Modo estritamente causal: depende apenas de amostras passadas x[n], x[n-1], x[n-2].
        filtered = sosfilt(sos, emg_array, axis=0)

    return filtered.astype(np.float32, copy=False)


def compute_causal_window_bounds(
    n_emg_samples: int,
    n_angle_frames: int,
    window_samples: int,
    trim_seconds: float = 5.0,
    emg_fs_hz: float = EMG_SAMPLE_RATE_HZ,
    kinematics_fs_hz: float = KINEMATICS_SAMPLE_RATE_HZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula à mão (hand-made) os índices exatos de sincronismo EMG <-> Vicon.

    Por que escrever essa matemática na mão em vez de reamostrar (`scipy.signal.resample`)?
    Reamostrar o EMG de 2052.52 Hz para 2000 Hz (ou interpolar os ângulos de 100 Hz
    para 2052.52 Hz) introduz artefatos de interpolação/ringing de Fourier e exige
    custo computacional desnecessário no sistema embarcado.
    Em vez disso, mantemos os dois relógios originais intactos e calculamos, para
    cada instante `t_k = k / 100.0` da câmera, o índice inteiro exato da amostra
    mais recente de EMG:
        n_end(k)   = round(t_k * 2052.52)
        n_start(k) = n_end(k) - W

    Retorna
    -------
    emg_start_idx : np.ndarray (int32), shape (n_valid_windows,)
        Índice inicial (inclusivo) de cada janela no array de EMG.
    emg_end_idx : np.ndarray (int32), shape (n_valid_windows,)
        Índice final (exclusivo) de cada janela no array de EMG.
    valid_frames : np.ndarray (int32), shape (n_valid_windows,)
        Índice do frame correspondente na tabela de ângulos do Vicon.
    """
    # 1. Determinar o primeiro e o último frame da câmera dentro de [trim, T_total - trim)
    first_frame = int(np.ceil(trim_seconds * kinematics_fs_hz))
    total_duration_s = n_angle_frames / kinematics_fs_hz
    last_frame = int(np.floor((total_duration_s - trim_seconds) * kinematics_fs_hz))

    if last_frame <= first_frame:
        raise ValueError(
            f"Intervalo inválido após trim de {trim_seconds}s "
            f"(first_frame={first_frame}, last_frame={last_frame})."
        )

    frame_indices = np.arange(first_frame, last_frame, dtype=np.int32)

    # 2. Instante físico (em segundos) de cada captura da câmera (100 Hz -> passos de 10 ms)
    label_times_s = frame_indices.astype(np.float64) / kinematics_fs_hz

    # 3. Conversão para o índice de amostra do amplificador EMG (2052.52 Hz)
    emg_end_idx = np.rint(label_times_s * emg_fs_hz).astype(np.int32)
    emg_start_idx = emg_end_idx - int(window_samples)

    # 4. Guarda de fronteira física (garante que nenhuma janela leia fora do buffer)
    valid_mask = (emg_start_idx >= 0) & (emg_end_idx <= n_emg_samples)

    return (
        emg_start_idx[valid_mask],
        emg_end_idx[valid_mask],
        frame_indices[valid_mask],
    )


def make_causal_windows_multidof(
    emg_array: np.ndarray,
    angles_df: pd.DataFrame,
    cfg: SignalProcessingConfig = SignalProcessingConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orquestra condicionamento + fatiamento causal vetorizado para 1 gravação.

    Parâmetros
    ----------
    emg_array : np.ndarray, shape (n_emg_samples, 128)
        Sinal bruto de HD-sEMG carregado pelo `io.load_emg_array`.
    angles_df : pd.DataFrame, shape (n_frames, 17)
        DataFrame de ângulos carregado pelo `io.load_angles_dataframe`.
    cfg : SignalProcessingConfig
        Configuração imutável (contendo `zero_phase=True` para referência de
        software ou `zero_phase=False` para simulação causal de hardware).

    Retorna
    -------
    windows : np.ndarray, shape (n_windows, window_samples, 128)
        Tensor 3D `float32` das janelas causais filtradas.
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

    # Etapa 1: Filtragem passa-alta na série contínua inteira (evita transiente de borda nas janelas)
    if cfg.highpass_hz is not None and cfg.highpass_hz > 0.0:
        emg_clean = apply_highpass_filter(
            emg_array,
            cutoff_hz=cfg.highpass_hz,
            order=cfg.filter_order,
            zero_phase=cfg.zero_phase,
        )
    else:
        emg_clean = emg_array

    # Etapa 2: Cálculo hand-made dos ponteiros de início/fim de cada janela causal
    window_samples = cfg.window_samples
    emg_start_idx, _, valid_frames = compute_causal_window_bounds(
        n_emg_samples=len(emg_clean),
        n_angle_frames=len(angles_df),
        window_samples=window_samples,
        trim_seconds=cfg.trim_seconds,
    )

    # Etapa 3: Construção da matriz de endereçamento 2D (Broadcasting NumPy sem loop Python)
    # emg_start_idx[:, None] tem shape (N, 1)
    # offsets[None, :]       tem shape (1, W) -> [0, 1, 2, ..., W-1]
    # A soma produz uma tabela de índices (N, W) onde a linha i contém [start_i, ..., end_i - 1].
    offsets = np.arange(window_samples, dtype=np.int32)[None, :]
    sample_idx_grid = emg_start_idx[:, None] + offsets

    windows = emg_clean[sample_idx_grid]  # (N, W, 128) em float32

    # Etapa 4: Extração simultânea dos 5 ângulos (Multi-DoF)
    target_matrix = angles_df.loc[:, list(cfg.target_dofs)].to_numpy(dtype=np.float32)
    y_angles = target_matrix[valid_frames]  # (N, 5) em float32

    return windows, y_angles, valid_frames
