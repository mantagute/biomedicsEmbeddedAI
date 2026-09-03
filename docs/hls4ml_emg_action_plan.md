# Plano de estudo: HD-sEMG, ML e hls4ml

## Direcao geral

O objetivo deste mes nao e estudar todos os modelos de ML para EMG. O objetivo e construir uma trilha curta e replicavel para transformar um sinal continuo de HD-sEMG em uma inferencia embarcavel em FPGA, usando hls4ml quando fizer sentido e Vitis/Vivado HLS para entender os relatorios e restricoes de hardware.

Para o escopo atual, o caminho mais eficiente e:

1. Entender o dataset e montar janelas sincronizadas EMG -> cinematica.
2. Treinar modelos pequenos e previsiveis.
3. Quantizar/comprimir esses modelos.
4. Converter para hls4ml e medir latencia, recursos e erro numerico.
5. Entender apenas o HLS necessario para interpretar e melhorar esses resultados.

## Decisoes ja fixadas pela documentacao local

- **Tarefa principal:** regressao continua de cinematica dos dedos a partir de HD-sEMG, nao classificacao discreta de gestos. A propria caracterizacao do dataset registra movimento sinusoidal continuo e ground truth cinematico continuo.
- **Entrada real do sistema:** somente HD-sEMG. A camera/Vicon entra como fonte de label durante coleta, treino e avaliacao offline; ela nao entra no sistema embarcado.
- **Hardware alvo:** comparacao entre placas Xilinx/AMD Zynq disponiveis, especialmente Zybo Z7 como caso restrito e ZCU104 como caso com folga de recursos. O tutorial local tambem usa `xc7z020clg484-1`, compativel com PYNQ-Z1/ZedBoard e proximo da classe Zybo Z7-20.
- **Arquitetura de implantacao:** fluxo PS + PL. O modelo convertido por hls4ml vira IP na PL, conectado por AXI/AXI-Stream, FIFO/DMA/ComBlock quando necessario, e controlado pelo PS/PYNQ durante bring-up.
- **Pipeline de processamento:** no primeiro ciclo, manter carregamento de dados, normalizacao, janelamento, avaliacao e parte do pre/post-processamento em Python/PS para iterar rapido. A inferencia pesada deve migrar para PL. Depois, mover features/pre-processamento causal para HLS/PL conforme a pressao de latencia e recursos, sobretudo no Zybo.
- **Latencia/cadencia:** a documentacao local usa a ideia de janela causal `T_a` e tempo de estimacao `T_e`. Para o dataset alvo, a cinematica esta a 100 Hz, entao a cadencia inicial de saida deve ser compativel com 100 Hz, isto e, passo de aproximadamente 10 ms quando quisermos predicao por amostra cinematica. O criterio de hardware e `T_e < T_a` para permitir sobreposicao de janelas; o limite de experiencia de controle deve ficar abaixo da faixa de centenas de milissegundos, idealmente preservando a janela recomendada de 150 ms como primeiro alvo.

## Dataset alvo

Dataset: HD sEMG of Forearm Muscles and 3D Hand Kinematics During Sinusoidally-Modulated Finger Movements and Grasping Tasks.

Caracteristicas importantes:

- 21 participantes saudaveis.
- HD-sEMG dos musculos EDC e FDS.
- Cinematica 3D da mao capturada simultaneamente, usada como label/ground truth.
- 8 tarefas motoras principais.
- Frequencias de movimento: 0.50 Hz e 0.75 Hz.
- 3 repeticoes por condicao.
- HD-sEMG amostrado em 2052.52 Hz.
- Cinematica amostrada em 100 Hz, definindo a cadencia natural inicial de labels/predicoes.
- Arquivos em CSV.
- Tamanho total aproximado: 18.07 GB.

Implicacao pratica: o primeiro entregavel deve usar apenas um subconjunto pequeno, por exemplo 1 participante, 1 tarefa, 1 frequencia e 1 repeticao. A prioridade e validar sincronizacao, janelamento, labels e formato de entrada do modelo.

## Modelos a priorizar

### 1. Baseline embarcavel: features temporais/espaciais + regressao leve

Entrada:

- Janelas causais de EMG, com 150 ms como alvo inicial documentado.
- Features por canal ou por grupo de canais: RMS, MAV, WL e descritores espaciais simples.
- Evolucao natural: MLD-BFM em blocos 2x2 quando o pipeline basico estiver validado.

Modelo:

- Ridge/Lasso como baseline minimo e MLP pequeno, por exemplo `Dense(64) -> Dense(32) -> output`.
- Saida continua: angulo/posicao cinematica de um ou mais DoFs.

Por que estudar:

- Excelente para primeiro fluxo hls4ml.
- Facil de quantizar.
- Relatorios de recurso ficam legiveis.
- Serve como baseline contra modelos espaciais/temporais.

### 2. Modelo direto leve: CNN pequena

Entrada:

- Janela bruta ou pre-processada de EMG.
- Representacao 1D temporal ou 2D espacial sobre as grades 8x8 de EDC/FDS.
- Opcional: reduzir canais antes, usando selecao espacial, blocos 2x2 ou agregacao.

Modelo:

- `Conv1D/Conv2D -> ReLU -> Flatten/GlobalAveragePooling -> Dense`.

Por que estudar:

- Aprende padroes locais no tempo e/ou na topologia espacial HD-sEMG.
- Mais compativel com hls4ml que LSTM.
- Bom meio-termo entre desempenho e implementabilidade, especialmente na ZCU104.

### 3. Modelo temporal candidato: TCN pequena

Entrada:

- Sequencia de janelas ou janela maior.

Modelo:

- Conv1D causal/dilatada pequena, sem atencao no primeiro ciclo.

Por que estudar:

- Pode capturar dependencia temporal melhor que CNN simples.
- Evita parte da complexidade de LSTM/GRU em FPGA.
- So deve entrar depois que MLP e CNN estiverem funcionando.

### Modelos a deixar para depois

- LSTM/GRU: relevantes para EMG, mas mais trabalhosos em hls4ml e mais caros em latencia/recursos.
- CNN-LSTM: bom como referencia academica, mas pesado para primeiro mes.
- Transformers/attention: interessantes para artigos, ruins como primeiro alvo de FPGA.
- GNN: promissor para topologia HD-sEMG, mas adiciona complexidade cedo demais.

## Plano de 4 semanas

### Semana 1: dataset e baseline offline

Entregaveis:

- Baixar ou mapear a estrutura do dataset.
- Criar notebook de inspecao: arquivos, colunas, canais, taxa de amostragem, plots curtos.
- Fixar o primeiro alvo de regressao: um DoF/finger-angle simples, usando a cinematica como label.
- Sincronizar EMG 2052.52 Hz com cinemática 100 Hz.
- Criar janelas de EMG e labels alinhados.
- Treinar baseline de regressao leve com features temporais/espaciais simples.

Meta de fim da semana:

- Um notebook que pega CSVs reais e gera `X_train`, `y_train`, `X_val`, `y_val`.
- Um baseline Ridge/Lasso ou MLP pequeno treinando e avaliando erro continuo.

### Semana 2: hls4ml minimo e quantizacao

Entregaveis:

- Reproduzir notebooks locais `P0_Dataset`, `P1_TF_Training`, `P2_NN_Pruning`, `P3_Compilation`.
- Converter o MLP baseline para hls4ml.
- Comparar predicao Python/TensorFlow vs hls4ml.
- Ajustar `Precision`, `ReuseFactor`, `Strategy` e `ClockPeriod`.
- Gerar primeiro relatorio de sintese.

Meta de fim da semana:

- Primeiro projeto hls4ml sintetizavel com modelo pequeno.
- Tabela: erro numerico, latencia estimada, DSP, LUT, FF, BRAM.

### Semana 3: CNN/TCN e leitura de HLS

Entregaveis:

- Treinar uma 1D CNN pequena no mesmo subset.
- Testar uma TCN muito pequena se a CNN estiver estavel.
- Converter pelo menos a CNN para hls4ml.
- Estudar pragmas HLS na pratica: `pipeline`, `unroll`, `array_partition`, `dataflow`, interfaces.
- Entender como `ReuseFactor` se traduz em paralelismo/recursos.

Meta de fim da semana:

- Comparacao regressao leve vs CNN em erro, latencia e recursos.

### Semana 4: consolidacao para o projeto real

Entregaveis:

- Expandir para mais participantes/tarefas.
- Definir protocolo de validacao: intra-sujeito, inter-sujeito ou leave-one-subject-out.
- Definir modelo candidato oficial para o projeto.
- Documentar pipeline reprodutivel.
- Listar riscos tecnicos: tamanho do modelo, BRAM/DSP, sincronizacao, drift entre sessoes, generalizacao.

Meta de fim da semana:

- Um mini-relatorio com decisao tecnica: modelo, entrada, janela, cadencia de predicao, alvo Zybo/ZCU104 e proximos experimentos.

## Conteudos priorizados

Esta lista e organizada pelo que desbloqueia implementacao. A regra e simples: consumir o conteudo, reproduzir um artefato pequeno, e so entao passar para o proximo.

### Checklist MVP, em ordem

Use este formato: marque `[x]` quando terminar, preencha a data, a evidencia e uma observacao curta. Evidencia pode ser notebook, script, commit, pasta gerada, tabela ou print do relatorio.

#### 1. Dataset e problema real

- [ ] **Consumir:** Figshare `HD sEMG of Forearm Muscles and 3D Hand Kinematics...`
- [ ] **Extrair:** estrutura dos CSVs, EMG 2052.52 Hz, cinematica 100 Hz, tarefas e repeticoes.
- [ ] **Entregar:** notebook que lista arquivos, carrega 1 EMG + 1 cinematica e plota 5 s.
- Data:
- Evidencia:
- Observacoes:

#### 2. Pipeline EMG -> label

- [ ] **Consumir:** `docs/emgMachineLearningLiteratureReview.md`, secoes S.0, S.0.1, S.2, S.3 e Ref. 17.
- [ ] **Extrair:** regressao continua, Vicon como label, janela 150 ms, MLD-BFM 2x2 como evolucao.
- [ ] **Entregar:** funcoes `make_windows()` e alinhamento EMG -> `y` cinematica.
- Data:
- Evidencia:
- Observacoes:

#### 3. Baseline mais simples

- [ ] **Consumir:** `docs/MLandDLFromScratch.md`, parte de regressao vs classificacao.
- [ ] **Extrair:** saida continua, regressao multi-output como proximo passo.
- [ ] **Entregar:** Ridge/Lasso com RMS/MAV/WL em 1 DoF.
- Data:
- Evidencia:
- Observacoes:

#### 4. Primeiro modelo neural

- [ ] **Consumir:** `hls4ml_tutorial/pyTraining/P1_TF_Training.ipynb`.
- [ ] **Extrair:** padrao de treino, split, normalizacao, salvamento de modelo.
- [ ] **Entregar:** MLP pequeno adaptado para regressao do dataset.
- Data:
- Evidencia:
- Observacoes:

#### 5. Primeiro hls4ml

- [ ] **Consumir:** hls4ml Setup and Quick Start.
- [ ] **Extrair:** `config_from_keras_model`, `convert_from_keras_model`, `compile`, `predict`, `build`.
- [ ] **Entregar:** MLP convertido para hls4ml, com comparacao Python/TensorFlow vs hls4ml.
- Data:
- Evidencia:
- Observacoes:

#### 6. Knobs de FPGA em hls4ml

- [ ] **Consumir:** hls4ml Concepts + Configuration.
- [ ] **Extrair:** `Precision`, `ReuseFactor`, `Strategy`, `IOType`, fixed point.
- [ ] **Entregar:** 3 sinteses variando precisao/reuse e tabela de recursos.
- Data:
- Evidencia:
- Observacoes:

#### 7. Quantizacao sem exagero

- [ ] **Consumir:** `hls4ml_tutorial/pyTraining/P2_NN_Pruning.ipynb`.
- [ ] **Assistir:** MIT EfficientML `Quantization (Part I)` e `Quantization (Part II)`.
- [ ] **Extrair:** por que reduzir bits antes de sintetizar; impacto no erro e nos recursos.
- [ ] **Entregar:** modelo quantizado simples e queda de erro medida.
- Data:
- Evidencia:
- Observacoes:

#### 8. CNN pequena

- [ ] **Consumir:** hls4ml tutorial de modelos/CNN + artigo de HD-sEMG/CNN.
- [ ] **Extrair:** representacao da janela como sequencia temporal ou grade 8x8.
- [ ] **Entregar:** CNN pequena comparada com MLP em erro e recursos.
- Data:
- Evidencia:
- Observacoes:

#### 9. HLS minimo para ler relatorio

- [ ] **Consumir:** AMD Vitis HLS docs de pragmas e otimizacao.
- [ ] **Extrair:** `pipeline`, `unroll`, `array_partition`, `dataflow`, gargalo de BRAM/DSP.
- [ ] **Entregar:** explicacao curta do relatorio de latencia/DSP/LUT/BRAM do seu modelo.
- Data:
- Evidencia:
- Observacoes:

#### 10. PS/PL e dados ate o IP

- [ ] **Consumir:** PYNQ DMA tutorial Part 1 e Part 2.
- [ ] **Extrair:** AXI DMA, AXI-Stream, buffers contiguos, `.bit` + `.hwh`.
- [ ] **Entregar:** loopback DMA entendido antes de conectar o IP do modelo.
- Data:
- Evidencia:
- Observacoes:

### Links essenciais

- Dataset Figshare: https://figshare.com/articles/dataset/HD_sEMG_of_Forearm_Muscles_and_3D_Hand_Kinematics_During_Sinusoidally-Modulated_Finger_Movements_and_Grasping_Tasks/31032934
- hls4ml docs: https://fastmachinelearning.org/hls4ml/
- hls4ml Setup and Quick Start: https://fastmachinelearning.org/hls4ml/intro/setup.html
- hls4ml Concepts: https://fastmachinelearning.org/hls4ml/api/concepts.html
- hls4ml tutorial notebooks: https://github.com/fastmachinelearning/hls4ml-tutorial
- AMD Vitis HLS: https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vitis/vitis-hls.html
- AMD Vitis HLS optimization/pragmas: https://docs.amd.com/r/en-US/ug1399-vitis-hls/Optimizing-Techniques-and-Troubleshooting-Tips
- Xilinx Vitis-HLS Introductory Examples: https://github.com/Xilinx/Vitis-HLS-Introductory-Examples
- PYNQ DMA Part 1, hardware design: https://discuss.pynq.io/t/tutorial-pynq-dma-part-1-hardware-design/3133
- PYNQ DMA Part 2, using DMA from PYNQ: https://discuss.pynq.io/t/tutorial-pynq-dma-part-2-using-the-dma-from-pynq/3134/
- MIT EfficientML/TinyML, curso com videos: https://hanlab.mit.edu/courses/2024-fall-65940

### Videos para assistir, sem dispersar

Assista videos apenas nestes pontos:

1. **Antes da quantizacao:** MIT EfficientML `Quantization (Part I)` e `Quantization (Part II)`.
2. **Antes de pruning:** MIT EfficientML `Pruning and Sparsity (Part I)`; Part II so se o pruning virar necessario.
3. **Antes de mexer em HLS manual:** AMD webinars/tutorials de Vitis HLS focados em pragmas/performance.
4. **Antes de board bring-up:** PYNQ DMA Part 1/2; se houver video equivalente, usar como apoio, mas o tutorial escrito e suficiente.

Nao assistir agora:

- NAS, LLM compression, transformers, attention, GNN.
- Cursos completos de FPGA do zero.
- Videos longos de Verilog/VHDL antes de ter o IP hls4ml funcionando.

### Artigos para ler com objetivo pratico

| Artigo | Quando ler | Ler para responder |
|---|---|---|
| Molinari & Elias, `Do Spatial Descriptors Improve Multi-DoF Finger Movement Decoding from HD sEMG?` | Semana 1 | Qual janela, quais features e por que regressao continua? |
| SEEDS dataset paper | Semana 1 ou 2 | Como outros organizam EMG + cinematica e validacao? |
| `Learning a Hand Model From Dynamic Movements Using High-Density EMG and Convolutional Neural Networks` | Semana 3 | Como usar HD-sEMG como estrutura espacial para CNN? |
| `Enhancing sEMG-Based Finger Motion Prediction with CNN-LSTM Regressors...` | So depois do MVP | O que CNN-LSTM ganha, e por que talvez seja pesado demais agora? |
| hls4ml paper/overview | Semana 2 | Como justificar hls4ml como ferramenta de inferencia low-latency em FPGA? |
| QKeras/hls4ml quantization paper | Semana 2 ou 3 | Como justificar quantizacao heterogenea/fixed-point? |

### Regra de corte para este mes

Se um conteudo nao ajuda a produzir uma destas quatro coisas, fica para depois:

1. `X, y` janelado e alinhado.
2. Baseline Ridge/MLP treinado.
3. Modelo convertido para hls4ml.
4. Relatorio de recursos/latencia entendido.

## Checklist de notebooks

- [ ] Rodar `hls4ml_tutorial/pyTraining/P0_Dataset.ipynb`.
- [ ] Rodar `hls4ml_tutorial/pyTraining/P1_TF_Training.ipynb`.
- [ ] Rodar `hls4ml_tutorial/pyTraining/P3_Compilation.ipynb`.
- [ ] Rodar `hls4ml_tutorial/pyTraining/P2_NN_Pruning.ipynb`.
- [ ] Criar um novo notebook para o dataset HD-sEMG com o mesmo formato mental dos notebooks acima.

Notas:

## Primeira implementacao no dataset

Escopo minimo:

- Participante: 1.
- Tarefa: flexao/extensao de indicador como primeiro alvo; pinça indicador-polegar entra depois por envolver polegar, que a documentacao aponta como mais dificil.
- Frequencia: 0.50 Hz.
- Repeticao: 1.
- Janela EMG: 150 ms como primeiro alvo; testar 100 ms e 200 ms se necessario.
- Passo: 10 ms para alinhar com a cinematica a 100 Hz; 20 ms pode ser usado como reducao de carga.
- Label: amostra cinemática mais proxima ao centro/fim da janela.
- Saida inicial: 1 variavel cinematica; depois 3 variaveis/DoFs, nao a mao inteira.

Experimentos:

- [ ] Features + Ridge/Lasso.
- [ ] Features + MLP.
- [ ] Janela bruta/reduzida + CNN pequena.
- [ ] Quantizacao e conversao hls4ml do melhor modelo simples.

Notas:

## Criterios de decisao

Um modelo so avanca se:

- [ ] A entrada e saida estao claramente definidas.
- [ ] O erro offline e aceitavel para o primeiro subset.
- [ ] A diferenca Keras vs hls4ml e pequena.
- [ ] O relatorio HLS cabe nos recursos da placa avaliada: Zybo Z7 para restricao dura, ZCU104 para folga.
- [ ] A latencia estimada respeita `T_e < T_a` e permite saida compativel com a cinematica a 100 Hz quando usado passo de 10 ms.

## Pendencias reais

- [ ] Confirmar se o primeiro ciclo de sintese sera feito para Zybo Z7-20/`xc7z020clg484-1` ou diretamente para ZCU104.
- [ ] Definir quais colunas cinematicas do CSV serao usadas no primeiro alvo de regressao.
- [ ] Decidir se a primeira implementacao hls4ml vai receber features ja calculadas no PS/Python ou uma janela EMG reduzida para a propria PL processar.
