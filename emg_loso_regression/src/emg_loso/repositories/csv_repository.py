"""
repositories/csv_repository.py
------------------------------
Camada de Armazenamento (Repository): Descoberta e Leitura dos CSVs Brutos do Dataset.

Responsabilidades:
1. Interpretar o nome do arquivo (`SubXXX_<Task>_<Freq>_450_<Rep>.csv`) devolvendo
   a entidade validada `RecordingInfo`.
2. Varrer a pasta de um sujeito no disco local ou SSD externo (`discover_subject_recordings`),
   pareando apenas gravações dinâmicas que possuam ambos os arquivos (EMG e Ângulos).
3. Carregar matrizes em memória com `float32` e validação de invariantes (`InvalidSignalShapeError`).
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from emg_loso.config import (
    ALL_ANGLE_CSV_COLUMNS,
    KNOWN_EXCLUDED_RECORDINGS,
    N_EMG_CHANNELS,
)
from emg_loso.domain.entities import RecordingInfo
from emg_loso.domain.exceptions import InvalidSignalShapeError

_RECORDING_PATTERN = re.compile(
    r"^(?P<subject>Sub\d{3})_(?P<task>[1-8])_(?P<freq>05|075)_(?P<duration>\d+)_(?P<rep>[012])$"
)


def parse_recording_id(
    recording_id: str,
    emg_path: Path,
    angles_path: Path,
) -> RecordingInfo | None:
    """Faz o parse do identificador da gravação para a entidade `RecordingInfo`.

    Retorna `None` se for condição suplementar estática (`rest` ou `start_position`).
    """
    match = _RECORDING_PATTERN.match(recording_id)
    if match is None:
        return None

    return RecordingInfo(
        subject=match.group("subject"),
        task_id=int(match.group("task")),
        freq_code=match.group("freq"),
        rep=int(match.group("rep")),
        recording_id=recording_id,
        emg_path=emg_path,
        angles_path=angles_path,
    )


def discover_subject_recordings(
    raw_data_root: str | Path,
    subject: str,
    extra_excluded: set[str] | frozenset[str] = frozenset(),
) -> list[RecordingInfo]:
    """Varre a pasta de um sujeito e retorna todas as gravações válidas pareadas."""
    subject_dir = Path(raw_data_root) / subject
    emg_dir = subject_dir / "HD_sEMG"
    angles_dir = subject_dir / "HandKinematics" / "Angles"

    if not emg_dir.exists():
        raise FileNotFoundError(f"Diretório HD_sEMG não encontrado: {emg_dir}")
    if not angles_dir.exists():
        raise FileNotFoundError(f"Diretório Angles não encontrado: {angles_dir}")

    excluded = KNOWN_EXCLUDED_RECORDINGS | frozenset(extra_excluded)
    recordings: list[RecordingInfo] = []

    for emg_file in sorted(emg_dir.glob("*.csv")):
        rec_id = emg_file.stem
        if rec_id in excluded:
            continue

        angles_file = angles_dir / f"{rec_id}.csv"
        if not angles_file.exists():
            continue

        info = parse_recording_id(rec_id, emg_path=emg_file, angles_path=angles_file)
        if info is not None:
            recordings.append(info)

    return sorted(recordings)


def load_emg_array(emg_path: str | Path) -> np.ndarray:
    """Carrega um CSV de HD-sEMG diretamente como array NumPy float32 (amostras, 128)."""
    df = pd.read_csv(
        Path(emg_path),
        header=None,
        dtype=np.float32,
        skipinitialspace=True,
    )
    arr = df.to_numpy(dtype=np.float32, copy=False)
    if arr.ndim != 2 or arr.shape[1] != N_EMG_CHANNELS:
        raise InvalidSignalShapeError(
            f"Esperado {N_EMG_CHANNELS} canais de EMG em {emg_path}, "
            f"mas encontrado shape {arr.shape}."
        )
    return arr


def load_angles_dataframe(angles_path: str | Path) -> pd.DataFrame:
    """Carrega um CSV de ângulos do Vicon descartando as 5 linhas de cabeçalho."""
    return pd.read_csv(
        Path(angles_path),
        header=None,
        names=list(ALL_ANGLE_CSV_COLUMNS),
        skiprows=5,
        dtype=np.float32,
        skipinitialspace=True,
    )
