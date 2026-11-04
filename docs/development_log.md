# Diário de Desenvolvimento e Roadmap: HD-sEMG em FPGA (hls4ml)

Este documento centraliza o registro das implementações realizadas, as decisões arquiteturais (o "porquê") e o escopo exato dos próximos passos necessários para darmos o Produto Mínimo Viável (MVP) como concluído.

## 1. Histórico de Implementações

### Dia 1: 14 de Setembro de 2026
**Foco:** Construção da fundação de dados, alinhamento temporal e validação matemática de regressão.

**O que fizemos:**
* **Pipeline Reutilizável (`src/emg_hls4ml_mvp/`)**:
  * `data_loading.py`: Leitura estruturada dos arrays de HD-sEMG (2052Hz) e cinemática de dedos (100Hz).
  * `windowing.py`: Criação de janelas causais (150ms) de amostras de EMG alinhadas à frequência de 100Hz da câmera.
  * `features.py`: Extração matemática de sinais no domínio do tempo (RMS, MAV, WL).
  * `dataset.py`: Encapsulamento completo que pega o CSV bruto e cospe tensores `X` (features) e `y` (ângulos) perfeitamente alinhados para treino.
  * *Nota de refatoração:* O script isolado de predição linear (`run_baseline.py`) foi movido para dentro de `src/` mantendo a limpeza do repositório.
* **Notebooks de Exploração e Treino (`notebooks/`)**:
  * `01_inspect_sub001.ipynb`: Validação dos dados brutos.
  * `02_baseline_regression.ipynb`: Treinamento de um Ridge Regression clássico (sem shuffle) que atingiu **R² ~0.89** provando que o sinal de entrada consegue prever as curvas do dedo.
  * `03_keras_and_hls4ml.ipynb`: Migração do baseline linear para uma pequena Rede Neural Multicamadas em TensorFlow/Keras (`Dense(32) -> ReLU -> Dense(1)`). Adicionado o fluxo inicial de tradução automática dessa rede para código C++ de hardware usando a biblioteca `hls4ml`.

**Por que fizemos isso:**
Para fazer o deploy em um chip (FPGA), nós não podemos enviar código Python. Precisamos traduzir a inteligência para um Circuito Digital (portas lógicas RTL e DSPs). A ferramenta capaz de fazer essa ponte é o `hls4ml`, porém ela só aceita Redes Neurais (Keras/PyTorch), não aceita modelos tradicionais estatísticos (como Ridge do scikit-learn). 
Portanto, foi necessário construir uma rede neural simples no Keras emulando o mesmo comportamento matemático do baseline. Outro motivo importante: validamos tudo offline no Mac primeiro porque compilar e debuggar direto na placa/simulador FPGA de hardware (Vivado) é extremamente lento; precisávamos ter certeza de que o dado original fazia sentido e gerava correlação antes de envolver circuitos integrados na jogada.

---

## 2. Próximos Passos (Trilha de Conclusão do MVP)

O MVP estará 100% validado quando o modelo neural que já treinamos gerar um IP de hardware sintetizável, que respeite os limites físicos (LUTs e Latência) de placas como Zybo Z7 ou ZCU104.

### Passo 1: Execução da Síntese em Ambiente Linux
* **Ação:** O processo parou no Mac devido à incompatibilidade do compilador da Apple Clang com os cabeçalhos aritméticos da Xilinx (erro de ambiguidade no `complex.h`). 
* **Execução:** Mover o projeto para um servidor Linux contendo o compilador do `Vivado HLS`.
* **Entrega:** Executar o `hls_model.build(csim=False, synth=True, vsynth=True)` no Notebook 03 e capturar os três relatórios cruciais: **Latência (Clock Cycles)**, **DSP48Es utilizados** e **LUTs gastas**.

### Passo 2: Sintonia de Quantização (Fixed-Point Tuning)
* **Ação:** Avaliar o relatório do Passo 1. FPGAs processam melhor pontos fixos inteiros (`<8,2>`, `<12,4>`) do que o padrão que vem no framework (`<16,6>`).
* **Execução:** Reduzir iterativamente os bits no `hls4ml_config` e executar simulações (`C-Sim`).
* **Entrega:** Encontrar o equilíbrio ótimo, conhecido como fronteira de Pareto: o mínimo absoluto de portas lógicas (tamanho do circuito) que mantém o nosso erro R² acima de ~0.85 (ou seja, que a predição continue suave e precisa).

### Passo 3: Escalonamento do Input (Modelos Espaciais ou Multi-DoF)
* **Ação:** Atualizar a prova de conceito de "1 Grau de Liberdade (Flexão do Indicador)" para alvos mais complexos, como movimento simultâneo de indicador e polegar (pinça).
* **Execução:** Caso as métricas base temporais falhem neste nível, substituir o MLP base por uma CNN minúscula que leia o HD-sEMG não como array, mas como "imagem topológica 2D", implementando filtros para as grades de eletrodos (EDC e FDS).
* **Entrega:** Repetir a síntese Linux (Passo 1 e 2) no novo modelo e validar se ainda é compatível com placas de entrada da classe do Zybo.

---

## 3. Fase 2: Expansão Científica do Artigo — Regressão Multi-DoF com LOSO e Personalização

### Dia 2: 04 de Novembro de 2026
**Foco:** Planejamento arquitetural e criação da fundação do pacote final (`emg_loso_regression/`) para expandir o artigo de classificação rumo à regressão contínua dos 5 ângulos articulares (`thumb_z`, `index_z`, `middle_z`, `ring_z`, `little_z`) sob validação Leave-One-Subject-Out (LOSO) e adaptação específica por sujeito (Head Transfer Learning).

#### 3.1 Fundamentação Metodológica (Alinhamento com o Artigo e Literatura)
* **Por que LOSO Zero-Shot $\rightarrow$ LOSO Personalizado (e não Intra-Sujeito ou 80/20 fixo)?**
  * Com $N = 21$ participantes, um split fixo 80/20 de sujeitos deixaria apenas 4 pessoas no teste (alta variância amostral e perda de ~20% de diversidade anatômica no treino). O LOSO atua como validação cruzada máxima ($K = 21$ folds), treinando o extrator geral com 20 sujeitos (95,2% da população) e testando em cada um dos 21 como usuário inédito.
  * Como o EMG sofre forte variabilidade inter-sujeito (anatomia, tecido subcutâneo e posicionamento das grades $8 \times 8$), avaliamos o modelo em dois estágios no `rep2` do sujeito *held-out*: (i) **Zero-Shot** (sem calibração) e (ii) **Personalizado (Head TL)**, onde o extrator neural $f_\theta(X)$ permanece congelado (fixo em hardware na FPGA) e apenas a camada linear final ($\hat{\mathbf{y}} = W f_\theta(X) + \mathbf{b}$) é recalibrada usando `rep0` e `rep1` do usuário alvo.
* **Parâmetros de Condicionamento e Janelamento (Molinari & Elias, 2026 + Artigo Seção III-A):**
  * **Janela de 150 ms (~308 amostras):** Adotada conforme demonstrado por Molinari & Elias (2026, Ref. 17) como o ponto ótimo para regressão cinemática contínua neste dataset (evitando o achatamento de picos em movimentos senoidais de 0.75 Hz que ocorreria em janelas longas de 250 ms).
  * **Filtro Passa-Alta em 20 Hz + Intervalo 5–40 s:** Mantém estrita consistência com a Seção III-A do artigo e recomendações da ISEK/literatura (Guo et al., 2024), eliminando artefatos mecânicos de movimento de cabos (<20 Hz) e transientes de início/fim de gravação.
* **Arquitetura Alvo de Deep Learning (MLP Baseline vs. 2D-CNN Topográfica $8 \times 8$):**
  * Enquanto a 1D-CNN de classificação do artigo colapsa os 128 canais logo na entrada com uma convolução $1 \times 1$, a decodificação contínua de múltiplos dedos exige preservar os gradientes espaciais locais sobre os músculos extensor (EDC, canais 1–64) e flexor (FDS, canais 65–128). Por isso, estruturamos o pipeline para alimentar tanto um **MLP** (vetor plano `(384,)`) quanto uma **2D-CNN Topográfica Leve** (tensor espacial `(8, 8, 6)` correspondente às duas grades $8 \times 8$ em RMS, MAV e WL, fundamentada em Tam et al., 2020 e Molinari & Elias, 2026).

#### 3.2 Decisões de Engenharia de Software Implementadas (`emg_loso_regression/`)
1. **Isolamento do Pacote Final (`emg_loso_regression/`):**
   * *Decisão:* Manter `emg_hls4ml_mvp/` intacto como registro da prova de conceito inicial e estruturar o pacote definitivo segundo o padrão PEP 621 (`pyproject.toml` + layout `src/emg_loso/`).
   * *Alternativa descartada:* Modificar os scripts do MVP diretamente ou usar scripts soltos sem pacote instalável (`sys.path.insert`). Descartado para evitar acoplamento com testes preliminares de 1 sujeito/1 dedo.
2. **Fonte Única de Verdade (`src/emg_loso/config.py`):**
   * *Decisão:* Centralizar constantes físicas (`2052.52 Hz`, `100.0 Hz`, geometria $2 \times 8 \times 8$, os 5 DoFs no eixo Z, janela de `150.0 ms`, filtro de `20.0 Hz`, e a separação `rep0/rep1` para calibração vs. `rep2` para teste) usando tipagem `Final` e `@dataclass(frozen=True)`.
   * *Alternativa descartada:* Passar argumentos soltos ou usar dicionários mutáveis (`dict`) ou arquivos `.yaml` externos sem validação de tipo. `dataclass(frozen=True)` garante autocompletar na IDE, checagem estática e imutabilidade em tempo de execução (nenhum fold pode alterar parâmetros acidentalmente).
3. **Descoberta Automática e Pareamento de Gravações (`src/emg_loso/io.py`):**
   * *Decisão:* Implementar expressão regular (`_RECORDING_PATTERN`) e o dataclass `RecordingInfo` para fazer o parse estruturado dos nomes de arquivo (`SubXXX_<Task>_<Freq>_450_<Rep>.csv`), filtrando automaticamente condições estáticas (`rest`, `start_position`) e garantindo que apenas gravações com ambos os arquivos (EMG e Ângulos) entrem no pipeline (48 gravações dinâmicas no `Sub001`: 32 de calibração e 16 de teste). Leitura do CSV de EMG feita via parser C do `pandas` convertendo direto para `np.ndarray` (`float32`, sem cópia).
   * *Alternativa descartada:* `np.loadtxt` ou `np.genfromtxt` (até 10x mais lentos em CSVs de 70 MB) ou montar nomes de arquivo via concatenação manual de strings (quebra quando um sujeito tem uma gravação excluída como `Sub017_3_05_450_1`).

#### 3.3 Roteiro das Próximas Etapas (Fase 2)
* **Etapa 1 (Em andamento):** Finalizar módulo de condicionamento/janelamento Multi-DoF (`windowing.py`, janela de 150 ms + filtro passa-alta 20 Hz) e extração de features com remodelagem topográfica `(N, 8, 8, 6)` (`features.py`).
* **Etapa 2:** Agrupamento intra-sujeito (`rep0 + rep1` vs. `rep2`) e validação Multi-DoF completa no `Sub001`.
* **Etapa 3:** Script de ingestão a partir do SSD externo e geração de cache compacto `.npz` para os 21 sujeitos (respeitando `Kinematic_review.ods`).
* **Etapa 4:** Implementação dos modelos Deep Learning (MLP e 2D-CNN Topográfica com cabeça linear separável) e motor de treinamento/adaptação LOSO nos 21 sujeitos.
* **Etapa 5:** Extração dos parâmetros de análise ($R^2$, RMSE, NRMSE, MAE, $r$ de Pearson), geração das tabelas/figuras do artigo e síntese de hardware (`hls4ml`).
