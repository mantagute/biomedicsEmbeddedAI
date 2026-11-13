"""Camada `domain/`: Entidades imutáveis, taxonomias, exceções, contratos e física pura do sinal."""

from emg_loso.domain.entities import (
    FoldPartition,
    RecordingInfo,
    SignalProcessingConfig,
    SubjectDataset,
)
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
from emg_loso.domain.ports import (
    AdaptableRegressorPort,
    SubjectCacheRepositoryPort,
)

__all__ = [
    "ALLOWED_REPS_BY_ROLE",
    "AdaptableRegressorPort",
    "ContractViolationError",
    "DataLeakageViolationError",
    "DomainError",
    "FingerDoF",
    "FoldPartition",
    "FrequencyCode",
    "InvalidSignalShapeError",
    "MotorTask",
    "RecordingInfo",
    "SignalProcessingConfig",
    "SplitRole",
    "SubjectCacheRepositoryPort",
    "SubjectDataset",
]
