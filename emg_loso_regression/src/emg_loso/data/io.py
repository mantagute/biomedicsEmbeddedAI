"""
data/io.py
----------
Camada de Entrada/Saída (I/O) e Descoberta de Gravações do Dataset.

Responsabilidades:
1. Interpretar a convenção oficial de nomenclatura dos arquivos:
   `SubXXX_<Task>_<Freq>_450_<Rep>.csv` -> objeto imutável `RecordingInfo`.
2. Varrer o diretório de um sujeito (`Sub001` ... `Sub021`), pareando apenas
   gravações de tarefas motoras que possuam TANTO o arquivo de HD-sEMG quanto
   o arquivo de Ângulos Cinemáticos (descartando `rest`, `start_position` e
   gravações na lista de exclusão).
3. Carregar os CSVs brutos em memória com tipagem `float32` eficiente.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from emg_loso.config import (
    ALL_ANGLE_CSV_COLUMNS,
    CALIBRATION_REPS,
    KNOWN_EXCLUDED_RECORDINGS,
    MOTOR_TASKS,
    N_EMG_CHANNELS,
    TEST_REPS,
)

# Expressão regular oficial para gravações de tarefas motoras:
# Exemplo: "Sub001_1_05_450_0" -> subject="Sub001", task=1, freq="05", duration=450, rep=0
_RECORDING_PATTERN = re.compile(
    r"^(?P<subject>Sub\d{3})_(?P<task>[1-8])_(?P<freq>05|075)_(?P<duration>\d+)_(?P<rep>[012])$"
)


@dataclass(frozen=True, order=True)
class RecordingInfo:
    """Metadados estruturados de uma gravação individual validada."""
    subject: str
    task_id: int
    freq_code: str
    rep: int
    recording_id: str
    emg_path: Path
    angles_path: Path

    @property
    def task_name(self) -> str:
        """Nome legível da tarefa motora (ex: 'Index', 'Tripod')."""
        return MOTOR_TASKS[self.task_id]

    @property
    def frequency_hz(self) -> float:
        """Frequência do movimento senoidal em Hz (0.50 ou 0.75)."""
        return 0.50 if self.freq_code == "05" else 0.75

    @property
    def is_calibration(self) -> bool:
        """True se pertencer às repetições de calibração/adaptação (rep0 ou rep1)."""
        return self.rep in CALIBRATION_REPS

    @property
    def is_test(self) -> bool:
        """True se pertencer à repetição reservada para teste (rep2)."""
        return self.rep in TEST_REPS


def parse_recording_id(
    recording_id: str,
    emg_path: Path,
    angles_path: Path,
) -> RecordingInfo | None:
    """Faz o parse do identificador da gravação.

    Retorna `None` caso o arquivo seja uma condição suplementar (`rest` ou
    `start_position`), pois estas não fazem parte das 8 tarefas dinâmicas.
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
    """Varre a pasta de um sujeito e retorna todas as gravações válidas pareadas.

    Parâmetros
    ----------
    raw_data_root : str | Path
        Diretório que contém as pastas `Sub001/`, `Sub002/`, etc.
        (Pode ser `data/raw` local ou o caminho direto no SSD externo).
    subject : str
        Identificador do sujeito, ex: `"Sub001"`.
    extra_excluded : set[str] | frozenset[str]
        Conjunto opcional de IDs adicionais a excluir (ex: vindos do Kinematic_review.ods).

    Retorna
    -------
    list[RecordingInfo]
        Lista ordenada por (task_id, freq_code, rep) contendo todas as gravações
        que possuem tanto CSV de EMG quanto CSV de Ângulos válidos.
    """
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
    """Carrega um CSV de HD-sEMG diretamente como array NumPy float32 (amostras, 128).

    Nota de engenharia:
    Usamos `pd.read_csv(..., dtype=np.float32).to_numpy()` porque o parser C do
    pandas é ~10x mais rápido que `np.loadtxt` para CSVs grandes (70 MB), e já
    devolvemos um `np.ndarray` enxuto sem overhead de índices do DataFrame.
    """
    df = pd.read_csv(
        Path(emg_path),
        header=None,
        dtype=np.float32,
        skipinitialspace=True,
    )
    arr = df.to_numpy(dtype=np.float32, copy=False)
    if arr.shape[1] != N_EMG_CHANNELS:
        raise ValueError(
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
