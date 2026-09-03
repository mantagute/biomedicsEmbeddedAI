# Semana 1 - Fundamentacao do pipeline EMG -> cinematica

Este documento explica a construcao conceitual do primeiro pipeline, sem depender de codigo pronto. A ideia e entender quais blocos precisam existir, por que eles existem e como cada um se relaciona com o tratamento de sinais EMG.

## 1. Qual e o problema computacional?

O dataset tem dois sinais sincronizados:

- **Entrada:** HD-sEMG, um sinal eletrico muscular multicanal.
- **Saida/label:** cinematica 3D da mao, medida por camera/Vicon.

Como a saida e continua, o problema principal e **regressao**, nao classificacao. O modelo deve aprender algo do tipo:

```text
janela recente de EMG -> posicao/angulo cinematico naquele instante
```

No sistema real, a camera nao existe. Ela serve apenas para ensinar e avaliar o modelo offline.

## 2. Por que usar janelas?

EMG bruto e um sinal oscilatorio, ruidoso e nao estacionario. Uma amostra isolada quase nunca diz algo confiavel sobre o movimento. Por isso, analisamos um intervalo curto de tempo.

No seu caso, a documentacao aponta **150 ms** como janela inicial forte.

Com a taxa de EMG do dataset:

```text
2052.52 amostras/s * 0.150 s ~= 308 amostras
```

Entao uma janela de 150 ms contem aproximadamente 308 amostras por canal.

Se existirem 128 canais, a janela bruta teria:

```text
308 amostras x 128 canais
```

Isso pode ser usado diretamente por uma CNN, mas para o MVP e melhor primeiro comprimir essa janela em features.

## 3. Por que alinhar com a cinematica?

A cinematica esta a **100 Hz**, ou seja:

```text
1 amostra a cada 10 ms
```

O EMG esta a **2052.52 Hz**, ou seja, muito mais rapido. Isso cria um problema de alinhamento:

```text
muitas amostras EMG -> uma amostra cinematica
```

A estrategia MVP e:

1. Pegar uma janela EMG causal.
2. Escolher o tempo de referencia da janela.
3. Associar essa janela a amostra cinematica mais proxima.

Existem duas escolhas comuns:

- **Label no fim da janela:** mais causal para tempo real. Usa o EMG acumulado ate agora para prever o estado atual.
- **Label no centro da janela:** pode funcionar melhor offline, mas usa uma interpretacao menos direta para tempo real.

Para FPGA/tempo real, comece com **label no fim da janela**.

## 4. O que significa janela causal?

Uma janela causal usa apenas passado e presente.

Em um instante `t`, voce pode usar:

```text
EMG[t - 150 ms ... t]
```

Voce nao pode usar:

```text
EMG[t - 75 ms ... t + 75 ms]
```

porque isso exigiria amostras futuras. Offline isso e possivel, mas em hardware real nao.

Esse ponto e central para HLS/FPGA: o sistema precisa operar como fluxo continuo, nao como uma analise depois que a gravacao inteira terminou.

## 5. O que sao features temporais?

Features temporais resumem uma janela EMG em poucos numeros. Elas reduzem o custo computacional e ajudam o primeiro modelo a ser simples.

### RMS

Mede energia/amplitude efetiva do sinal na janela.

Intuicao:

```text
quanto maior a ativacao muscular, maior tende a ser o RMS
```

E bom para EMG porque a amplitude do sinal carrega informacao sobre contracao muscular.

### MAV

Media do valor absoluto.

Intuicao:

```text
intensidade media da atividade muscular
```

E mais barato que RMS, pois evita raiz quadrada.

### WL

Waveform Length soma quanto o sinal variou ao longo da janela.

Intuicao:

```text
quanto mais o sinal oscila, maior a WL
```

Ela captura atividade dinamica, nao apenas amplitude.

## 6. Como fica a forma dos dados?

Imagine uma janela com:

```text
308 amostras x 128 canais
```

Se voce calcula RMS, MAV e WL para cada canal:

```text
128 RMS + 128 MAV + 128 WL = 384 features
```

Entao cada janela vira uma linha:

```text
X[i] = 384 numeros
y[i] = angulo/posicao cinematica correspondente
```

Para comecar, use apenas um alvo cinematico:

```text
y[i] = uma coluna da cinematica
```

Depois voce expande para multi-output:

```text
y[i] = varios angulos/DoFs
```

## 7. Por que Ridge/Lasso antes de MLP?

Antes de treinar uma rede neural, e util ter um baseline linear.

Ridge/Lasso responde:

```text
features simples ja explicam alguma parte da cinematica?
```

Se o baseline linear for completamente ruim, pode haver problema em:

- alinhamento temporal,
- escolha do alvo cinematico,
- normalizacao,
- canais usados,
- tarefa escolhida,
- janelamento.

Ou seja: ele e um teste de sanidade antes de culpar o modelo.

## 8. Por que MLP depois?

O MLP pequeno e o primeiro modelo neural ideal porque:

- recebe features ja compactas;
- e facil de treinar;
- e facil de converter para hls4ml;
- gera relatorios HLS mais faceis de entender;
- permite testar quantizacao sem muita complexidade.

O papel dele nao e ser o modelo final. O papel dele e validar o fluxo:

```text
dataset -> features -> modelo -> hls4ml -> relatorio
```

## 9. Onde entram CNNs?

CNNs entram quando voce quiser explorar melhor a estrutura espacial/temporal do HD-sEMG.

Existem duas representacoes naturais:

### CNN 1D temporal

Cada canal e uma serie temporal. A CNN aprende padroes locais ao longo do tempo.

Boa pergunta:

```text
o formato da onda nos ultimos 150 ms ajuda mais que RMS/MAV/WL?
```

### CNN 2D espacial

Cada grade 8x8 pode ser tratada como uma imagem muscular. A CNN aprende regioes de ativacao.

Boa pergunta:

```text
a distribuicao espacial da ativacao muscular ajuda a prever o movimento?
```

Para o MVP, isso vem depois do baseline com features.

## 10. O que precisa ser implementado primeiro?

Nao pense ainda em hls4ml. Primeiro construa estes blocos mentalmente e depois em notebook:

### Bloco A - Inspecao

Perguntas que esse bloco responde:

- Quantos arquivos existem?
- Quais parecem ser EMG?
- Quais parecem ser cinematica?
- Quantas colunas cada CSV tem?
- Existem colunas de tempo ou o tempo precisa ser reconstruido pela taxa de amostragem?

### Bloco B - Escolha do subconjunto

Escolha minima:

- 1 participante;
- 1 tarefa;
- 0.50 Hz;
- 1 repeticao;
- 1 arquivo EMG;
- 1 arquivo cinematico.

Isso evita que problemas de escala escondam problemas conceituais.

### Bloco C - Reconstrucao de tempo

Se o CSV nao tiver tempo explicito:

```text
tempo_emg[n] = n / 2052.52
tempo_cinematica[k] = k / 100
```

Isso permite alinhar amostras dos dois sinais.

### Bloco D - Janelamento

Para cada janela:

- inicio;
- fim;
- tempo de referencia;
- trecho EMG correspondente;
- label cinematico mais proximo.

Exemplo conceitual:

```text
janela EMG de 0.000 s ate 0.150 s -> label cinematico em 0.150 s
janela EMG de 0.010 s ate 0.160 s -> label cinematico em 0.160 s
janela EMG de 0.020 s ate 0.170 s -> label cinematico em 0.170 s
```

Isso gera uma predicao a cada 10 ms, compatível com 100 Hz.

### Bloco E - Extracao de features

Para cada janela e cada canal:

- RMS;
- MAV;
- WL.

Resultado:

```text
uma matriz X com uma linha por janela
```

### Bloco F - Baseline

Treine Ridge/Lasso ou MLP pequeno.

Metricas iniciais:

- RMSE;
- R2;
- correlacao;
- grafico predicao vs ground truth.

## 11. Como isso conversa com FPGA?

Cada decisao acima tem uma traducao para hardware:

| Decisao no pipeline | Significado em FPGA |
|---|---|
| Janela de 150 ms | buffer circular de amostras |
| Passo de 10 ms | nova predicao a cada amostra cinematica |
| RMS/MAV/WL | operacoes aritmeticas simples, boas para HLS |
| MLP pequeno | MACs em DSP blocks |
| Quantizacao | menos BRAM/DSP/LUT |
| `T_e < T_a` | hardware processa a janela antes de acumular a proxima |

O motivo de comecar com features e MLP e que esse caminho gera um primeiro IP simples. Depois voce decide se vale migrar mais preprocessamento para PL.

## 12. Ordem recomendada de construcao

- [ ] Entender a estrutura dos CSVs.
- [ ] Escolher um par EMG + cinematica.
- [ ] Reconstruir ou ler vetores de tempo.
- [ ] Plotar 5 segundos de EMG e cinematica.
- [ ] Criar janelas causais de 150 ms.
- [ ] Associar cada janela a uma label cinematica.
- [ ] Calcular RMS/MAV/WL.
- [ ] Formar `X` e `y`.
- [ ] Treinar Ridge/Lasso em 1 alvo.
- [ ] Plotar predicao vs ground truth.
- [ ] So entao partir para MLP.

## 13. Principio guia

O primeiro MVP nao precisa ser bonito. Ele precisa provar que:

```text
um trecho real de HD-sEMG consegue virar uma matriz X,
uma coluna real de cinematica consegue virar y,
e um modelo simples aprende alguma relacao entre os dois.
```

Se isso estiver funcionando, hls4ml passa a ser uma etapa de engenharia. Se isso nao estiver funcionando, hls4ml so vai acelerar um pipeline errado.
