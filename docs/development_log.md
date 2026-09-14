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
