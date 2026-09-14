"""
dataset.py
----------
Pipeline de construção do dataset para uma gravação.

Este módulo é o ponto de entrada para os notebooks e scripts.
Ele orquestra o fluxo completo:

    CSV (disco) → DataFrame (pandas) → janelas (NumPy) → features (NumPy)

Entrada: identificadores de sujeito e gravação + parâmetros de janela.
Saída  : X (features), y (labels cinemáticos), label_indices (posição temporal).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from emg_hls4ml_mvp.data_loading import load_angles_csv, load_emg_csv, recording_paths
from emg_hls4ml_mvp.features import extract_time_domain_features
from emg_hls4ml_mvp.windowing import make_causal_windows


def build_feature_dataset(
    data_root: str | Path,
    subject: str,
    recording_id: str,
    target_column: str = "index_z",
    window_ms: float = 150.0,
    trim_seconds: float = 5.0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constrói X e y para uma gravação a partir dos CSVs brutos.

    Parâmetros
    ----------
    data_root : str | Path
        Raiz do diretório de dados (contém a subpasta raw/).
    subject : str
        Identificador do sujeito, ex: "Sub001".
    recording_id : str
        Identificador da gravação, ex: "Sub001_1_05_450_0".
    target_column : str
        Coluna do CSV de ângulos usada como label de regressão.
    window_ms : float
        Duração da janela causal de EMG em milissegundos.
    trim_seconds : float
        Segundos descartados nas bordas do sinal.
    verbose : bool
        Se True, imprime shapes intermediários para validação do pipeline.

    Retorna
    -------
    X : np.ndarray, shape (n_windows, n_features)
        Matriz de features. Cada linha é uma janela de EMG representada
        pela concatenação de RMS, MAV e WL por canal.
    y : np.ndarray, shape (n_windows,)
        Vetor de labels (valor do ângulo cinemático em cada janela).
    label_indices : np.ndarray, shape (n_windows,)
        Índices dos frames cinemáticos — útil para reconstruir a série temporal.
    """
    # --- Etapa 1: leitura dos CSVs brutos ---
    emg_path, angles_path = recording_paths(data_root, subject, recording_id)
    emg    = load_emg_csv(emg_path)
    angles = load_angles_csv(angles_path)

    if verbose:
        print(f"EMG carregado    : {emg.shape}    (amostras × canais)")
        print(f"Ângulos carregados: {angles.shape} (frames × DoFs)")

    # --- Etapa 2: janelamento e alinhamento temporal ---
    # Cada janela causal de EMG é alinhada ao frame cinemático correspondente.
    windows, y, label_indices = make_causal_windows(
        emg,
        angles,
        target_column=target_column,
        window_ms=window_ms,
        trim_seconds=trim_seconds,
    )

    if verbose:
        print(f"Janelas geradas  : {windows.shape} (janelas × amostras × canais)")

    # --- Etapa 3: extração de features ---
    # Reduz cada janela 2D (amostras × canais) em um vetor 1D de features.
    X = extract_time_domain_features(windows)

    if verbose:
        print(f"Features extraídas: {X.shape} (janelas × features)")
        print(f"Labels             : {y.shape}")

    return X, y, label_indices
