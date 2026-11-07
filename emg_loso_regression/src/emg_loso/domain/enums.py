"""Taxonomias de estado, tarefas motoras e papéis de particionamento do domínio."""
from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Final


class MotorTask(IntEnum):
    """As 8 tarefas motoras dinâmicas do protocolo experimental (Molinari & Elias, 2026)."""

    INDEX = 1
    MIDDLE = 2
    RING_LITTLE = 3
    THUMB = 4
    INDEX_PINCH = 5
    MIDDLE_PINCH = 6
    TRIPOD = 7
    FIVE_FINGER = 8

    @property
    def label(self) -> str:
        """Nome legível da tarefa motora para relatórios e gráficos."""
        return _TASK_LABELS[self]


_TASK_LABELS: Final[dict[MotorTask, str]] = {
    MotorTask.INDEX: "Index",
    MotorTask.MIDDLE: "Middle",
    MotorTask.RING_LITTLE: "Ring-Little",
    MotorTask.THUMB: "Thumb",
    MotorTask.INDEX_PINCH: "Index Pinch",
    MotorTask.MIDDLE_PINCH: "Middle Pinch",
    MotorTask.TRIPOD: "Tripod",
    MotorTask.FIVE_FINGER: "5-Finger",
}


class FrequencyCode(StrEnum):
    """Código da frequência da senoide guia no nome do arquivo CSV."""

    HZ_0_50 = "05"
    HZ_0_75 = "075"

    @property
    def hz(self) -> float:
        """Valor numérico da frequência em Hertz."""
        return 0.50 if self == FrequencyCode.HZ_0_50 else 0.75


class FingerDoF(StrEnum):
    """Os 5 Graus de Liberdade (DoFs) efetivos da mão no eixo Z do Vicon."""

    THUMB_Z = "thumb_z"
    INDEX_Z = "index_z"
    MIDDLE_Z = "middle_z"
    RING_Z = "ring_z"
    LITTLE_Z = "little_z"


class SplitRole(StrEnum):
    """Papel da repetição dentro do protocolo LOSO + Personalização."""

    CALIBRATION = "CALIBRATION"
    TEST = "TEST"


# Matriz de transição de domínio: relaciona cada papel às únicas repetições permitidas.
ALLOWED_REPS_BY_ROLE: Final[dict[SplitRole, frozenset[int]]] = {
    SplitRole.CALIBRATION: frozenset({0, 1}),
    SplitRole.TEST: frozenset({2}),
}
