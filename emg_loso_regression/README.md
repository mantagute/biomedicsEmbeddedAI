# HD-sEMG Personalized Kinematic Regression (LOSO & FPGA Pipeline)

Motor de condicionamento de sinais, extração topográfica e regressão contínua de ângulos articulares dos 5 dedos (`thumb_z`, `index_z`, `middle_z`, `ring_z`, `little_z`) a partir de gravações de **HD-sEMG (128 canais, duas matrizes 8×8 sobre os músculos EDC e FDS)** sob o protocolo **Leave-One-Subject-Out (LOSO)** com **Personalização por Transfer Learning (Head Adaptation)**.

> **Especificação Técnica (RFC):** Detalhes sobre o desenho do sistema, contratos de camadas, matriz de transição anti-*leakage*, fluxos de execução e *trade-offs* arquiteturais estão documentados na **[RFC-0001 (docs/ARCHITECTURE.md)](docs/ARCHITECTURE.md)**.

---

## Arquitetura do Serviço (Padrão Base-Service + Pipes & Filters + Microkernel)

O pacote combina o isolamento em camadas **Base-Service** (`domain/`, `controllers/`, `connectors/`, `repositories/`) com **Pipes & Filters** para o condicionamento matemático de sinais (`domain/signal/`) e **Microkernel** para o acoplamento de redes neurais:

| Camada | Pacote | Responsabilidade |
| :--- | :--- | :--- |
| **`domain/`** | `src/emg_loso/domain/` | Entidades imutáveis com invariantes no `__post_init__` (`slots=True, frozen=True`), enumeradores (`SplitRole`, `MotorTask`, `FingerDoF`) e exceções (`ContractViolationError`, `InvalidSignalShapeError`, `DataLeakageViolationError`). |
| **`domain/signal/`** | `src/emg_loso/domain/signal/` | **Estágios Pipes & Filters puros (0 I/O):** `filtering.py` (Butterworth SOS 20 Hz), `windowing.py` (sincronismo $2052.52\text{ Hz} \leftrightarrow 100\text{ Hz}$ e fatiamento causal de 150 ms), `features.py` (`RMS`, `MAV`, `WL` normalizado) e `topography.py` (planejado). |
| **`repositories/`** | `src/emg_loso/repositories/` | **Camada de Armazenamento:** `csv_repository.py` (descoberta via Regex das 48 gravações por sujeito e leitura `float32`) e `cache_repository.py` (planejado para persistência `.npz`). |
| **`connectors/`** | `src/emg_loso/connectors/` | **Plugins do Microkernel (Planejados):** Adaptadores para `DeepMLP`, `TopographicCNN2D` e síntese `hls4ml`. |
| **`controllers/`** | `src/emg_loso/controllers/` | **Núcleo do Microkernel (Planejado):** Orquestração dos 21 folds do LOSO (*Zero-Shot* vs. *Personalizado*) e cálculo de métricas. |

---

## Estrutura Atual de Arquivos

```text
emg_loso_regression/
├── docs/
│   └── ARCHITECTURE.md                    # [RFC-0001] Especificação arquitetural completa
├── pyproject.toml                         # Empacotamento PEP 621
├── README.md                              # Visão geral do projeto
└── src/
    └── emg_loso/
        ├── config.py                      # [Concluído] Constantes físicas e geometria 8x8
        ├── domain/
        │   ├── enums.py                   # [Concluído] MotorTask, FrequencyCode, FingerDoF, SplitRole
        │   ├── exceptions.py              # [Concluído] DomainError, ContractViolationError, InvalidSignalShapeError
        │   ├── entities.py                # [Concluído] SignalProcessingConfig e RecordingInfo (com __post_init__)
        │   └── signal/
        │       ├── filtering.py           # [Concluído] Filter 1: Butterworth SOS 20 Hz (Zero-Phase / Causal)
        │       ├── windowing.py           # [Concluído] Filter 2: Sincronismo e janelamento causal Multi-DoF
        │       ├── features.py            # [Concluído] Filter 3: Extração hand-made RMS, MAV, WL -> (N, 384)
        │       └── topography.py          # [Próximo]   Filter 4: Mapeamento espacial 2D -> (N, 8, 8, 6)
        ├── repositories/
        │   └── csv_repository.py          # [Concluído] Descoberta de gravações e leitura CSV float32
        ├── connectors/                    # [Planejado] Adaptadores de modelos DL e hls4ml
        └── controllers/                   # [Planejado] Orquestrador LOSO e métricas científicas
```
