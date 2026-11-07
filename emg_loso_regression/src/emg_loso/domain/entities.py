"""Entidades e objetos de valor do domínio.

Escrito apenas com biblioteca padrão do Python: sem acoplamento a frameworks de IA ou disco.
Cada classe garante suas próprias invariantes em `__post_init__` antes de existir em memória.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from emg_loso.config import (
    CALIBRATION_REPS,
    EMG_SAMPLE_RATE_HZ,
    FINGER_DOF_COLUMNS,
    MOTOR_TASKS,
    TEST_REPS,
)
from emg_loso.domain.enums import ALLOWED_REPS_BY_ROLE, SplitRole
from emg_loso.domain.exceptions import ContractViolationError


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
