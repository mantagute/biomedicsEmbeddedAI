"""
config.py
---------
Fonte única de verdade (Single Source of Truth) para as constantes físicas
do equipamento de aquisição, topologia das matrizes de eletrodos e regras do
protocolo experimental (Molinari & Elias, 2026).

Por que este módulo existe?
Em projetos de pesquisa aplicada e sistemas embarcados, espalhar "números mágicos"
(ex: 2052.52 Hz, 128 canais, 5 segundos de corte, nomes de colunas) dentro de
funções gera inconsistências silenciosas entre os experimentos e o texto do artigo.
Centralizar tudo em estruturas imutáveis (`frozen=True`) garante reprodutibilidade.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# 1. Constantes Físicas de Aquisição (Imutáveis do Hardware de Coleta)
# ---------------------------------------------------------------------------

# Frequência de amostragem do amplificador de HD-sEMG (Hz).
EMG_SAMPLE_RATE_HZ: Final[float] = 2052.52

# Frequência de captura óptica de movimento (câmeras Vicon Nexus) (Hz).
KINEMATICS_SAMPLE_RATE_HZ: Final[float] = 100.0

# Geometria física dos arrays de eletrodos:
# Duas grades de 8x8 (64 canais no músculo extensor EDC + 64 no flexor FDS).
N_GRIDS: Final[int] = 2
GRID_ROWS: Final[int] = 8
GRID_COLS: Final[int] = 8
CHANNELS_PER_GRID: Final[int] = GRID_ROWS * GRID_COLS  # 64
N_EMG_CHANNELS: Final[int] = N_GRIDS * CHANNELS_PER_GRID  # 128


# ---------------------------------------------------------------------------
# 2. Definição das Colunas Cinemáticas (Vicon Angles CSV)
# ---------------------------------------------------------------------------

# Todas as 17 colunas presentes nos arquivos CSV de HandKinematics/Angles/.
ALL_ANGLE_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "frame",
    "sub_frame",
    "thumb_x",  "thumb_y",  "thumb_z",
    "index_x",  "index_y",  "index_z",
    "middle_x", "middle_y", "middle_z",
    "ring_x",   "ring_y",   "ring_z",
    "little_x", "little_y", "little_z",
)

# Os 5 Graus de Liberdade (DoFs) efetivos da mão.
# No modelo biomecânico do dataset, os eixos X e Y são fixados em 0.0,
# e toda a excursão angular (flexão/extensão nas articulações MCP e
# oposição/reposição na articulação CMC do polegar) ocorre no eixo Z.
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

# Mapeamento oficial das 8 tarefas motoras do dataset (1 a 8).
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

# Códigos de frequência de modulação senoidal nos nomes de arquivo ("05" = 0.50 Hz, "075" = 0.75 Hz).
FREQUENCY_CODES: Final[tuple[str, ...]] = ("05", "075")

# Divisão oficial de repetições do protocolo LOSO + Personalização do artigo:
# - rep0 e rep1: usados exclusivamente para calibração/adaptação (Personalization).
# - rep2: reservado exclusivamente para avaliação final (Held-out Test).
CALIBRATION_REPS: Final[tuple[int, ...]] = (0, 1)
TEST_REPS: Final[tuple[int, ...]] = (2,)

# Gravações sabidamente ausentes ou excluídas na documentação oficial do dataset (README.md):
KNOWN_EXCLUDED_RECORDINGS: Final[frozenset[str]] = frozenset({
    "Sub017_3_05_450_1",  # Excluída oficialmente no dataset original
})


# ---------------------------------------------------------------------------
# 4. Configuração Parametrizável do Pipeline de Processamento
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalProcessingConfig:
    """Parâmetros de janelamento e condicionamento de sinal.

    Usamos `@dataclass(frozen=True)` para impedir que qualquer função altere
    acidentalmente os parâmetros no meio de um experimento LOSO.
    """
    window_ms: float = 150.0
    trim_seconds: float = 5.0
    highpass_hz: float | None = 20.0
    filter_order: int = 4
    target_dofs: tuple[str, ...] = FINGER_DOF_COLUMNS

    @property
    def window_samples(self) -> int:
        """Converte a duração da janela (ms) em número inteiro de amostras de EMG."""
        return int(round(EMG_SAMPLE_RATE_HZ * self.window_ms / 1000.0))
