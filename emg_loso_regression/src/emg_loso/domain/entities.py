"""Entidades e objetos de valor do domínio.

Escrito apenas com biblioteca padrão do Python + NumPy: sem acoplamento a frameworks de IA ou disco.
Cada classe garante suas próprias invariantes em `__post_init__` antes de existir em memória.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from emg_loso.config import (
    CALIBRATION_REPS,
    EMG_SAMPLE_RATE_HZ,
    FINGER_DOF_COLUMNS,
    MOTOR_TASKS,
    N_DOFS,
    N_EMG_CHANNELS,
    TEST_REPS,
)
from emg_loso.domain.enums import ALLOWED_REPS_BY_ROLE, SplitRole
from emg_loso.domain.exceptions import (
    ContractViolationError,
    DataLeakageViolationError,
    InvalidSignalShapeError,
)


@dataclass(frozen=True, slots=True)
class SignalProcessingConfig:
    """Configuração imutável de condicionamento de sinal e janelamento causal.

    Valida suas próprias invariantes físicas na criação (`__post_init__`) e expõe
    dois presets formais:
    1. `software_reference()` (`zero_phase=True`): atraso de grupo zero (`sosfiltfilt`).
    2. `hardware_causal()` (`zero_phase=False`): filtragem causal estrita (`sosfilt`).
    """

    window_ms: float = 150.0
    trim_seconds: float = 5.0
    highpass_hz: float | None = 20.0
    filter_order: int = 4
    zero_phase: bool = True
    target_dofs: tuple[str, ...] = FINGER_DOF_COLUMNS

    def __post_init__(self) -> None:
        if self.window_ms <= 0.0:
            raise ContractViolationError(
                f"SignalProcessingConfig.window_ms must be > 0 (got {self.window_ms})."
            )
        if self.trim_seconds < 0.0:
            raise ContractViolationError(
                f"SignalProcessingConfig.trim_seconds must be >= 0 (got {self.trim_seconds})."
            )
        if self.highpass_hz is not None and self.highpass_hz > 0.0 and self.filter_order < 1:
            raise ContractViolationError(
                f"SignalProcessingConfig.filter_order must be >= 1 when highpass_hz is active "
                f"(got {self.filter_order})."
            )
        if not self.target_dofs:
            raise ContractViolationError("SignalProcessingConfig.target_dofs cannot be empty.")

        invalid_dofs = [d for d in self.target_dofs if d not in FINGER_DOF_COLUMNS]
        if invalid_dofs:
            raise ContractViolationError(
                f"Unknown target_dofs {invalid_dofs}. Allowed: {FINGER_DOF_COLUMNS}"
            )

    @property
    def window_samples(self) -> int:
        """Converte a duração da janela (ms) em número inteiro de amostras de EMG."""
        return int(round(EMG_SAMPLE_RATE_HZ * self.window_ms / 1000.0))

    @classmethod
    def software_reference(cls) -> SignalProcessingConfig:
        """Preset para benchmark offline em software (fase zero, sem atraso de grupo)."""
        return cls(zero_phase=True)

    @classmethod
    def hardware_causal(cls) -> SignalProcessingConfig:
        """Preset fiel ao pipeline físico em tempo real na FPGA (filtragem causal)."""
        return cls(zero_phase=False)


@dataclass(frozen=True, slots=True, order=True)
class RecordingInfo:
    """Entidade imutável que representa uma gravação pareada e validada de um sujeito."""

    subject: str
    task_id: int
    freq_code: str
    rep: int
    recording_id: str
    emg_path: Path
    angles_path: Path

    def __post_init__(self) -> None:
        if not self.subject or not self.subject.startswith("Sub"):
            raise ContractViolationError(
                f"Invalid RecordingInfo.subject: '{self.subject}'."
            )
        if self.task_id not in MOTOR_TASKS:
            raise ContractViolationError(
                f"Invalid RecordingInfo.task_id: {self.task_id}. Allowed: {list(MOTOR_TASKS)}"
            )
        if self.freq_code not in ("05", "075"):
            raise ContractViolationError(
                f"Invalid RecordingInfo.freq_code: '{self.freq_code}'. Allowed: ('05', '075')"
            )
        allowed_all_reps = CALIBRATION_REPS + TEST_REPS
        if self.rep not in allowed_all_reps:
            raise ContractViolationError(
                f"Invalid RecordingInfo.rep: {self.rep}. Allowed: {allowed_all_reps}"
            )

    @property
    def task_name(self) -> str:
        """Nome legível da tarefa motora (ex: 'Index', 'Tripod')."""
        return MOTOR_TASKS[self.task_id]

    @property
    def frequency_hz(self) -> float:
        """Frequência do movimento senoidal em Hz (0.50 ou 0.75)."""
        return 0.50 if self.freq_code == "05" else 0.75

    @property
    def split_role(self) -> SplitRole:
        """Papel formal da gravação segundo a matriz de transição da RFC-0001."""
        if self.rep in ALLOWED_REPS_BY_ROLE[SplitRole.CALIBRATION]:
            return SplitRole.CALIBRATION
        return SplitRole.TEST

    @property
    def is_calibration(self) -> bool:
        """True se pertencer às repetições de calibração/adaptação (rep0 ou rep1)."""
        return self.split_role == SplitRole.CALIBRATION

    @property
    def is_test(self) -> bool:
        """True se pertencer à repetição reservada para teste (rep2)."""
        return self.split_role == SplitRole.TEST


@dataclass(frozen=True, slots=True)
class SubjectDataset:
    """Dataset consolidado de 1 sujeito (todas as suas gravações válidas empilhadas).

    Armazena apenas `x_flat` (N, 384) para economizar RAM e disco; a grade 2D
    `(N, 8, 8, 6)` é derivada sob demanda em < 5 ms via `flat_to_topographic_grid`.
    """

    subject: str
    x_flat: np.ndarray
    y_angles: np.ndarray
    rep_ids: np.ndarray
    task_ids: np.ndarray
    freq_codes: np.ndarray

    def __post_init__(self) -> None:
        if not self.subject or not self.subject.startswith("Sub"):
            raise ContractViolationError(f"Invalid SubjectDataset.subject: '{self.subject}'.")

        n_samples = len(self.x_flat)
        if n_samples == 0:
            raise InvalidSignalShapeError(f"SubjectDataset '{self.subject}' cannot be empty.")

        expected_flat_dim = N_EMG_CHANNELS * 3  # 384
        if self.x_flat.ndim != 2 or self.x_flat.shape[1] != expected_flat_dim:
            raise InvalidSignalShapeError(
                f"Expected x_flat shape (N, {expected_flat_dim}), got {self.x_flat.shape}."
            )
        if self.y_angles.ndim != 2 or self.y_angles.shape != (n_samples, N_DOFS):
            raise InvalidSignalShapeError(
                f"Expected y_angles shape ({n_samples}, {N_DOFS}), got {self.y_angles.shape}."
            )
        for name, arr in (
            ("rep_ids", self.rep_ids),
            ("task_ids", self.task_ids),
            ("freq_codes", self.freq_codes),
        ):
            if arr.shape != (n_samples,):
                raise InvalidSignalShapeError(
                    f"Expected {name} shape ({n_samples},), got {arr.shape}."
                )

        unique_reps = set(np.unique(self.rep_ids).tolist())
        if not unique_reps.issubset({0, 1, 2}):
            raise ContractViolationError(
                f"Invalid rep_ids {unique_reps} in '{self.subject}'. Allowed: {{0, 1, 2}}."
            )

    @property
    def n_windows(self) -> int:
        return int(self.x_flat.shape[0])

    @property
    def calibration_mask(self) -> np.ndarray:
        """Máscara booleana selecionando exclusivamente `rep0` e `rep1`."""
        return np.isin(self.rep_ids, CALIBRATION_REPS)

    @property
    def test_mask(self) -> np.ndarray:
        """Máscara booleana selecionando exclusivamente `rep2`."""
        return np.isin(self.rep_ids, TEST_REPS)

    def get_calibration_split(self) -> tuple[np.ndarray, np.ndarray]:
        """Retorna `(x_flat_calib, y_angles_calib)` usando apenas `rep0 + rep1`."""
        mask = self.calibration_mask
        return self.x_flat[mask], self.y_angles[mask]

    def get_test_split(self) -> tuple[np.ndarray, np.ndarray]:
        """Retorna `(x_flat_test, y_angles_test)` usando apenas `rep2`."""
        mask = self.test_mask
        return self.x_flat[mask], self.y_angles[mask]


@dataclass(frozen=True, slots=True)
class FoldPartition:
    """Contrato imutável de particionamento de 1 Fold LOSO com barreira anti-leakage.

    Qualquer tentativa de incluir `target_subject` dentro de `source_subjects`,
    ou de misturar `rep2` nas repetições de calibração, dispara `DataLeakageViolationError`
    antes mesmo de iniciar o treinamento.
    """

    fold_index: int
    target_subject: str
    source_subjects: tuple[str, ...]
    calibration_reps: tuple[int, ...] = CALIBRATION_REPS
    test_reps: tuple[int, ...] = TEST_REPS

    def __post_init__(self) -> None:
        if self.fold_index < 0:
            raise ContractViolationError(
                f"FoldPartition.fold_index must be >= 0 (got {self.fold_index})."
            )
        if not self.source_subjects:
            raise ContractViolationError("FoldPartition.source_subjects cannot be empty.")
        if len(set(self.source_subjects)) != len(self.source_subjects):
            raise ContractViolationError("Duplicate subjects found in FoldPartition.source_subjects.")
        if self.target_subject in self.source_subjects:
            raise DataLeakageViolationError(
                f"Subject leakage detected: '{self.target_subject}' cannot appear in source_subjects."
            )
        if set(self.calibration_reps) & set(self.test_reps):
            raise DataLeakageViolationError(
                f"Repetition leakage detected: calibration_reps {self.calibration_reps} "
                f"and test_reps {self.test_reps} must be disjoint."
            )
        if not set(self.calibration_reps).issubset(ALLOWED_REPS_BY_ROLE[SplitRole.CALIBRATION]):
            raise DataLeakageViolationError(
                f"calibration_reps {self.calibration_reps} violates SplitRole.CALIBRATION "
                f"allowed reps {ALLOWED_REPS_BY_ROLE[SplitRole.CALIBRATION]}."
            )
        if not set(self.test_reps).issubset(ALLOWED_REPS_BY_ROLE[SplitRole.TEST]):
            raise DataLeakageViolationError(
                f"test_reps {self.test_reps} violates SplitRole.TEST "
                f"allowed reps {ALLOWED_REPS_BY_ROLE[SplitRole.TEST]}."
            )
