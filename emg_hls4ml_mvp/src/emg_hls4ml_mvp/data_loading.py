"""
data_loading.py
---------------
Funções de leitura de CSV do dataset HD-sEMG.

Responsabilidade: carregar arquivos brutos do disco e devolver DataFrames
pandas com colunas nomeadas. Pandas é usado apenas nesta camada de I/O —
a partir do windowing o dado vira NumPy.

Estrutura de arquivos esperada:
    <data_root>/raw/<subject>/HD_sEMG/<recording_id>.csv
    <data_root>/raw/<subject>/HandKinematics/Angles/<recording_id>.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Taxa de amostragem do equipamento HD-sEMG (Hz).
EMG_SAMPLE_RATE_HZ: float = 2052.52

# Taxa de amostragem do sistema de captura de cinemática (câmera Vicon) (Hz).
# Essa taxa define a cadência dos labels e a frequência de predição alvo.
KINEMATICS_SAMPLE_RATE_HZ: float = 100.0

# Nomes das colunas do CSV de ângulos exportado pelo Vicon.
# As 2 primeiras são metadados de frame; as 15 seguintes são ângulos dos dedos (x, y, z).
ANGLE_COLUMNS = [
    "frame",
    "sub_frame",
    "thumb_x", "thumb_y", "thumb_z",
    "index_x", "index_y", "index_z",
    "middle_x", "middle_y", "middle_z",
    "ring_x",   "ring_y",   "ring_z",
    "little_x", "little_y", "little_z",
]

# O dataset tem 128 eletrodos organizados em duas grades HD de 8x8 (EDC e FDS).
N_EMG_CHANNELS: int = 128


def load_emg_csv(path: str | Path) -> pd.DataFrame:
    """Carrega um CSV de HD-sEMG e devolve um DataFrame (linhas=tempo, colunas=canais).

    Os canais são nomeados ch_001 ... ch_128 para facilitar filtragem e
    inspeção sem precisar memorizar índices numéricos.
    """
    channel_names = [f"ch_{i:03d}" for i in range(1, N_EMG_CHANNELS + 1)]

    return pd.read_csv(
        Path(path),
        header=None,
        names=channel_names,
        dtype="float32",
        skipinitialspace=True,
    )


def load_angles_csv(path: str | Path) -> pd.DataFrame:
    """Carrega um CSV de ângulos do Vicon e devolve um DataFrame com colunas nomeadas.

    Os 5 primeiros linhas do CSV são cabeçalho do Vicon e são descartados.
    """
    return pd.read_csv(
        Path(path),
        header=None,
        names=ANGLE_COLUMNS,
        skiprows=5,
        dtype="float32",
        skipinitialspace=True,
    )


def recording_paths(
    data_root: str | Path,
    subject: str,
    recording_id: str,
) -> tuple[Path, Path]:
    """Devolve os caminhos de EMG e Ângulos para um ID de gravação.

    Levanta FileNotFoundError com mensagem clara se o arquivo não existir,
    evitando erros silenciosos de carregamento.
    """
    subject_dir = Path(data_root) / "raw" / subject
    filename = f"{recording_id}.csv"

    emg_path    = subject_dir / "HD_sEMG" / filename
    angles_path = subject_dir / "HandKinematics" / "Angles" / filename

    if not emg_path.exists():
        raise FileNotFoundError(f"Arquivo EMG não encontrado: {emg_path}")
    if not angles_path.exists():
        raise FileNotFoundError(f"Arquivo de ângulos não encontrado: {angles_path}")

    return emg_path, angles_path
