# HD-sEMG Personalized Kinematic Regression (LOSO & FPGA Pipeline)

Implementação estruturada em padrão de produção/pesquisa reprodutível para **regressão contínua de ângulos articulares dos 5 dedos** (`thumb_z`, `index_z`, `middle_z`, `ring_z`, `little_z`) a partir de sinais de **HD-sEMG (128 canais, duas matrizes 8×8 sobre os músculos EDC e FDS)** sob o protocolo **Leave-One-Subject-Out (LOSO)** com **Personalização por Transfer Learning (Head Adaptation)**.

Este pacote preserva o MVP original em `../emg_hls4ml_mvp/` intacto e organiza o entregável final do artigo em camadas desacopladas.

## Fundamentação Metodológica

1. **Protocolo LOSO ($N = 21$ sujeitos) — Zero-Shot vs. Personalizado:**
   * Em cada um dos 21 folds, o extrator neural ($f_\theta$) é treinado com os outros 20 sujeitos.
   * No sujeito *held-out*, as gravações `rep0` e `rep1` são reservadas para recalibrar apenas a camada linear de saída ($\hat{\mathbf{y}} = W f_\theta(X) + \mathbf{b}$), enquanto `rep2` é usada exclusivamente para avaliação cega (tanto em *Zero-Shot* quanto pós-*Personalização*).
2. **Condicionamento e Janelamento (Molinari & Elias, 2026 + Artigo Seção III-A):**
   * Filtro passa-alta Butterworth em **20 Hz** (remoção de artefatos mecânicos de movimento).
   * Recorte do intervalo estacionário **5–40 s** (`trim_seconds = 5.0`).
   * Janelas causais de **150 ms** (~308 amostras a 2052.52 Hz) alinhadas à captura óptica Vicon (100 Hz).
3. **Representação Espacial Topográfica ($8 \times 8 \times 6$):**
   * Extração de descritores temporais (**RMS, MAV, WL**) estruturados tanto em vetor plano `(384,)` (para baseline MLP) quanto em tensor topográfico 2D `(8, 8, 6)` (preservando a geometria física das duas grades $8 \times 8$ sobre EDC e FDS para a **2D-CNN Topográfica**).

## Arquitetura do Pacote (`src/emg_loso/`)

```text
emg_loso_regression/
├── pyproject.toml             # Empacotamento padrão PEP 621 (instalável via pip install -e .)
├── README.md                  # Visão geral e guia de arquitetura
└── src/
    └── emg_loso/
        ├── __init__.py
        ├── config.py          # [Concluído] Constantes físicas, protocolo LOSO e geometria 8x8
        ├── io.py              # [Concluído] Descoberta automática de gravações, regex parser e I/O float32
        ├── windowing.py       # [Concluído] Condicionamento (20 Hz Zero-Phase/Causal) e janelamento hand-made Multi-DoF
        ├── features.py        # [Próximo] Extração vetorial (RMS, MAV, WL) e tensor topográfico 2D (8x8x6)
        ├── models.py          # [Planejado] Redes Deep Learning (MLP e 2D-CNN Topográfica) com cabeça adaptável
        ├── loso.py            # [Planejado] Motor de avaliação LOSO (Zero-Shot vs. Personalized Head-TL)
        └── metrics.py         # [Planejado] Parâmetros de análise (RMSE, NRMSE, MAE, R², Pearson r)
```
