"""Camada `domain/`: Entidades imutáveis, taxonomias, exceções e física pura do sinal."""

from emg_loso.domain.entities import RecordingInfo, SignalProcessingConfig
from emg_loso.domain.enums import (
    ALLOWED_REPS_BY_ROLE,
    FingerDoF,
    FrequencyCode,
    MotorTask,
    SplitRole,
)
from emg_loso.domain.exceptions import (
    ContractViolationError,
    DataLeakageViolationError,
    DomainError,
    InvalidSignalShapeError,
)

__all__ = [
    "ALLOWED_REPS_BY_ROLE",
    "ContractViolationError",
    "DataLeakageViolationError",
    "DomainError",
    "FingerDoF",
    "FrequencyCode",
    "InvalidSignalShapeError",
    "MotorTask",
    "RecordingInfo",
    "SignalProcessingConfig",
    "SplitRole",
]
