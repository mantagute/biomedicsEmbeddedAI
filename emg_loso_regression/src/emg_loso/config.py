"""
config.py
---------
Fonte única de verdade (Single Source of Truth) para as constantes físicas
do equipamento de aquisição, topologia das matrizes de eletrodos e regras do
protocolo experimental (Molinari & Elias, 2026).
"""
from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# 1. Constantes Físicas de Aquisição (Imutáveis do Hardware de Coleta)
# ---------------------------------------------------------------------------

EMG_SAMPLE_RATE_HZ: Final[float] = 2052.52
KINEMATICS_SAMPLE_RATE_HZ: Final[float] = 100.0

N_GRIDS: Final[int] = 2
GRID_ROWS: Final[int] = 8
GRID_COLS: Final[int] = 8
CHANNELS_PER_GRID: Final[int] = GRID_ROWS * GRID_COLS  # 64
N_EMG_CHANNELS: Final[int] = N_GRIDS * CHANNELS_PER_GRID  # 128


# ---------------------------------------------------------------------------
# 2. Definição das Colunas Cinemáticas (Vicon Angles CSV)
# ---------------------------------------------------------------------------

ALL_ANGLE_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "frame",
    "sub_frame",
    "thumb_x",  "thumb_y",  "thumb_z",
    "index_x",  "index_y",  "index_z",
    "middle_x", "middle_y", "middle_z",
    "ring_x",   "ring_y",   "ring_z",
    "little_x", "little_y", "little_z",
)

FINGER_DOF_COLUMNS: Final[tuple[str, ...]] = (
    "thumb_z",
    "index_z",
    "middle_z",
    "ring_z",
    "little_z",
)
N_DOFS: Final[int] = len(FINGER_DOF_COLUMNS)  # 5


# ---------------------------------------------------------------------------
# 3. Protocolo Experimental (Tarefas, Frequências, Repetições e Exclusões)
# ---------------------------------------------------------------------------

MOTOR_TASKS: Final[dict[int, str]] = {
    1: "Index",
    2: "Middle",
    3: "Ring-Little",
    4: "Thumb",
    5: "Index Pinch",
    6: "Middle Pinch",
    7: "Tripod",
    8: "5-Finger",
}

FREQUENCY_CODES: Final[tuple[str, ...]] = ("05", "075")

CALIBRATION_REPS: Final[tuple[int, ...]] = (0, 1)
TEST_REPS: Final[tuple[int, ...]] = (2,)

KNOWN_EXCLUDED_RECORDINGS: Final[frozenset[str]] = frozenset({
    "Sub017_3_05_450_1",
})
