# EMG Signal Processing & Pattern Recognition
### Literature Reference Guide

---

## Contents

- [Reference 1 — Piyathilaka et al., 2026](#reference-1--piyathilaka-et-al-2026)
- [Reference 2 — Mendes et al.](#reference-2--mendes-et-al)
- [Reference 3 — Reaz et al., 2006](#reference-3--reaz-et-al-2006)
- [Reference 4 — Choi, 2023](#reference-4--choi-2023)
- [Reference 5 — Phinyomark et al., 2018](#reference-5--phinyomark-et-al-2018)
- [Reference 6 — Li et al., 2021](#reference-6--li-et-al-2021)
- [Reference 7 — Lima et al., 2024](#reference-7--lima-et-al-2024)
- [Reference 8 — Peres, 2016](#reference-8--peres-2016)
- [Reference 9 — Pérez-Reynoso et al., 2022](#reference-9--pérez-reynoso-et-al-2022)
- [Reference 10 — Fathi et al., 2026](#reference-10--fathi-et-al-2026)
- [Reference 11 — Souza, 2023](#reference-11--souza-2023)
- [Reference 12 — Xavier, 2021](#reference-12--xavier-2021)
- [Reference 13 — Guo et al., 2024](#reference-13--guo-et-al-2024)
- [Reference 14 — Tam et al., 2020](#reference-14--tam-et-al-2020)
- [Reference 15 — Lu et al., 2023](#reference-15--lu-et-al-2023)
- [Synthesis — From Literature to FPGA Implementation Strategy](#synthesis--from-literature-to-fpga-implementation-strategy)

---

## Reference 1 — Piyathilaka et al., 2026

L. Piyathilaka, J.-H. Sul, S. D. Arachchige, A. Jayawardena, and D. Moratuwage, "Advances in EMG Signal Processing and Pattern Recognition: Techniques, Challenges, and Emerging Applications," *Electronics*, vol. 15, no. 3, p. 590, 2026.

---

### 1.1 System & hardware trade-offs

| Topic | Description |
|---|---|
| Non-stationarity | sEMG pipelines degrade over time due to electrode shifts, sweat, and fatigue. |
| Hardware reality | HD-sEMG fixes spatial shifts but introduces heavy computational loads and energy drains, making it prohibitive for edge/embedded devices. |
| Classifiers | Traditional ML (SVM, LDA) is computationally cheap but vulnerable to signal variations. LSTM/GRU handle temporal dynamics much better. |

---

### 1.2 Signal processing pipeline

To make raw muscle data readable by a neural network without draining embedded resources, the signal must follow this strict conditioning flow.

| # | Stage | Description |
|:---:|---|---|
| — | **Raw sEMG signal** | Highly noisy, affected by skin impedance and sweat. |
| 1 | **Band-pass filtering** | High-pass (5–10 Hz) removes motion artifacts and baseline drift. Low-pass (400–450 Hz) suppresses electrical interference. |
| 2 | **Rectification & conditioning** | Translates negative wave amplitudes into positive values for easier mathematical analysis. |
| 3 | **Feature extraction** | Compresses the data stream into dense mathematical metrics across time, frequency, or non-linear domains. |
| 4 | **Classification / pattern recognition** | Features are fed into the ML or neural network model (e.g., LSTM). |

---

### 1.3 Feature extraction trade-off sheet

| Domain | Feature | Complexity | Noise Robustness | Real-Time | Key Justification |
|---|---|---|---|---|---|
| Time | `MAV` | Low | Low | Excellent | Cheapest to compute, but highly sensitive to electrode shifts and high-amplitude noise. |
| Time | `RMS` | Low | Medium | Excellent | Best balance of noise robustness and low cost; strong physiological relevance. |
| Time | `WL` | Low | Medium | Excellent | Captures amplitude variation; robust to scaling but requires solid preprocessing. |
| Frequency | `MNF` | Medium | High | Good | Tracks fatigue well, but highly sensitive to spectral outliers/noise. |
| Frequency | `MDF` | Medium | High | Good | Safer choice than MNF; less sensitive to spectral spikes. |
| Time-Freq. | `STFT` | High | High | Fair | Simple conceptually, but performs poorly in highly dynamic movements. |
| Time-Freq. | `CWT` | Very High | Very High | Poor | Excellent dynamic detail, but demands excessive parameter tuning and compute cost. |
| Time-Freq. | `DWT` | Med-High | High | Fair | Most efficient TFD method, but suffers from shift sensitivity and limited scale flexibility. |
| Non-linear | `SampEn` | Medium | Very High | Good | Preferred over ApEn for non-Gaussian EMG complexity; less sensitive to data length. |
| Non-linear | `ApEn` | Medium | High | Good | Handles complexity but is biased for short data segments. |

---

### 1.4 ML architecture trade-offs

| Architecture | Strengths | Limitations |
|---|---|---|
| Classical ML (SVM, LDA) | Fast, highly interpretable. | Weak user generalization; completely dependent on handcrafted features. |
| CNN | Strong spatial learning. | Computationally complex; sensitive to electrode shifts unless heavily augmented. |
| RNN (LSTM, GRU) | Strong temporal modeling. | Limited long-range context; higher inference latency and compute cost. |
| Hybrid CNN–RNN | Best spatio-temporal fusion. | Heavy computation; large parameter counts make embedded deployment difficult without pruning. |
| Transformers | Long-range attention. | Very high computational cost; unsuitable for real-time/battery-powered edge systems. |
| GNNs | Strong spatial topology. | Complex graph setup required. |
| Multimodal fusion | High robustness (integrates IMU/FMG). | Extra sensors required; increased sync complexity, cost, and power draw. |

---

### 1.5 Embedded constraints & resource footprint

| Model Family | Inference Latency | Memory Footprint | Power & Hardware Suitability |
|---|---|---|---|
| Classical ML (TD + LDA/SVM) | µs – low ms | ~1 kB | Excellent for MCU/DSP; guarantees long battery life. |
| Shallow NN / CNN | Low ms | 10–100 kB | Suitable for standard embedded edge devices. |
| CNN–RNN hybrids | Several ms – 10 ms | 100 kB – 1 MB | Feasible on edge hardware only with careful optimization and pruning. |
| Transformers / Attention | > 10 ms | MB range | Typically requires a dedicated Edge AI accelerator; drains battery quickly. |

---

### 1.6 Challenges & research gaps

- **Electrode shift & non-stationarity** — Signal structures alter drastically as sensors shift during daily activities, forcing frequent manual recalibration.
- **Physiological & session variability** — High inter-subject and intra-session variance driven by:
  - Fluctuations in skin impedance and subcutaneous fat layers
  - Variations in muscle fiber composition and changing force levels exerted during movements
  - Quality of the electrode–skin contact interface over time
- **Data constraints** — A persistent lack of standardized, large-scale, public EMG datasets restricts model training scaling, raising ethical, privacy, and data collection considerations.

---

### 1.7 Future trends & emerging solutions

| Paradigm | Emerging Solution | Engineering Value |
|---|---|---|
| On-device adaptation | Lightweight Edge AI with stable, incremental updating strategies. | Solves calibration-free operation by continuously fighting sensor drift without cloud dependencies. |
| Model optimization | Quantization, model compression, and structural pruning. | Directly addresses hardware barriers, adapting heavy networks down to wearable battery envelopes. |
| Generalization frameworks | Transfer learning and domain adaptation techniques. | Mitigates cross-session performance degradation with minimal user calibration data. |
| Multimodal fusion | Co-registering sEMG with IMU, FMG, or optical sensors. | Stabilizes gesture predictions under dynamic arm positions and muscle fatigue. |
| Data expansion | Synthetic EMG data generation frameworks. | Bypasses data scarcity and privacy bottlenecks by training on simulated physiological signals. |

---

## Reference 2 — Mendes et al.

J. Mendes et al., "Comparative Analysis Among Feature Selection of sEMG Signal for Hand Gesture Classification by Armband," *UTFPR*.

---

### 2.1 Usability vs. hardware constraints

- **The placement problem** — Precise muscle targeting yields high accuracy, but everyday users cannot accurately locate specific muscle groups.
- **The armband solution** — Uses fixed, equidistant sensor arrays. Shifts the engineering burden away from perfect physical placement and onto the processing algorithms.

---

### 2.2 Targeted preprocessing

- **6th-order Butterworth notch filter** — Explicitly used to eliminate 60 Hz powerline interference. The high-order approximation provides a razor-sharp cutoff slope, killing grid hum while preserving vital muscle data directly adjacent to it.

---

### 2.3 Sequential forward selection (SFS) wrapper

Tests features individually to find a winner, then iteratively tests combinations with remaining features to build an optimal subset.

| Approach | Engineering Pros | Engineering Cons |
|---|---|---|
| SFS wrapper | Tailors a highly accurate, custom feature subset optimized directly for your specific classifier. | High computational overhead during the initial offline training/search phase. |

---

### 2.4 Embedded optimization — LDA synergy

- **Dimensionality reduction** — Trimming the feature pool is mandatory before embedding code onto a microcontroller.
- **LDA boundary synergy** — Pairing highly correlated, mutually supportive features explicitly strengthens the mathematical decision boundaries of LDA, leading to cleaner cluster separation.
- **Bottom line** — Removing redundant attributes yields identical classification accuracy for core gestures at radically lower computational load on embedded hardware.

---

## Reference 3 — Reaz et al., 2006

M. B. I. Reaz, M. S. Hussain, and F. Mohd-Yasin, "Techniques of EMG Signal Analysis: Detection, Processing, Classification and Applications," *Biological Procedures Online*, vol. 8, no. 1, p. 11–35, 2006.

---

### 3.1 Physiological & mathematical nature of sEMG

- **Skeletal muscle focus** — EMG captures the bioelectric currents generated exclusively during the contraction of skeletal muscles.
- **The filtered impulse model** — The signal is mathematically modeled as a filtered impulse process where:
  - **The filter:** Motor Unit Action Potential (MUAP)
  - **The input:** Neuron firing pulses, modeled as a stochastic Poisson process

---

### 3.2 Signal integrity & noise categories

| Noise Type | Origin | Mitigation / Character |
|---|---|---|
| Inherent electronic | Semiconductor components in the amplifier/hardware. | Cannot be eliminated; requires high-quality hardware front-ends. |
| Ambient noise | Electromagnetic radiation from the surrounding environment (e.g., wall outlets). | Typically manifests as sharp electrical spikes. |
| Motion artifact | Relative movement between skin, electrode interface, and wires. | Creates low-frequency baseline fluctuations (< 10 Hz). |
| Inherent instability | Natural random firing behavior of active motor units. | Treated as a purely stochastic, non-deterministic property. |

 > ⚠️ **Critical engineering conflict — the notch filter debate:**
 > While Mendes et al. (Ref. 2) explicitly advocate for a 6th-order notch filter to kill 60 Hz hum, Reaz et al. explicitly warn that **notch filters are not recommended** because they strip away crucial physiological frequency components and distort raw signal peaks.
 > *Design choice:* Prefer high-pass filtering over notch filtering unless environmental line noise completely saturates the ADC.

---

### 3.3 Conditioning & segmentation foundations

- **Full-wave rectification** — Taking the absolute value of each data point is heavily preferred over half-wave rectification to preserve total signal energy for amplitude calculations.
- **On-off muscle detection** — The Lanyi and Adler method is cited as the benchmark for fast, reliable mathematical detection of rest-to-contraction transitions.
- **Non-stationarity realities** — Signal fluctuations happen at two speeds:
  - **Slow non-stationarity:** Caused by biochemical metabolite accumulation (muscle fatigue)
  - **Fast non-stationarity:** Driven by sudden changes in movement physics (dynamic biomechanics)

---

### 3.4 Advanced feature extraction techniques

- **Wavelet transform (WT)** — The ultimate mathematical tool for localized time-frequency analysis. Excels at breaking down fast, transient, and highly non-stationary muscle signals without losing time orientation.
- **Autoregressive (AR) time series** — Models the current sEMG output based on its delayed historical states. Acts as a powerful predictor when treating surface measurements as an extension of deep intramuscular signals.
- **Higher-order statistics (HOS)** — A probability-based expectation framework used when the signal deviates from a normal Gaussian distribution. Explicitly used to decode non-linearities and estimate signal phase information that standard linear tools miss.

---

### 3.5 Hardware real-time deployment realities

- **DSP multipliers** — Achieving real-time execution on embedded hardware depends heavily on hardware-level multipliers inside DSPs or FPGA slices to compute fast transforms (FFT/WT).
- **Fuzzy logic vs. neural networks** — While ANNs are ideal for adaptive pattern recognition, **fuzzy logic systems** offer a significant advantage when processing sEMG data because they are designed to handle highly inconsistent, unrepeatable, or contradictory biological signals.

---

### 3.6 Integrated circuit (IC) design constraints

Because raw EMG signals feature very low voltage amplitudes and carry low-frequency common-mode noise, hardware solutions (like those by Yen et al.) often require custom integrated chips.

- **System-on-Chip (SoC) integration** — Instrumentation amplifiers, gain control stages, and filters are integrated directly into a single processing chip.
- **Engineering goals** — This hardware-level integration achieves three critical benchmarks for embedded edge devices: low cost, low power consumption, and minimized physical layout area.

---

### 3.7 Methodological comparison matrix

*A rapid breakdown of the advantages and limitations of various EMG processing and classification methodologies.*

| Method | Advantages | Disadvantages & Limitations |
|---|---|---|
| **Double-threshold detection** | Higher detection probability than single-threshold; allows users to tune the trade-off between false alarms and detection probability. | — |
| **Wavelet Transform (WT)** | Linear, yielding multiresolution representation; unaffected by crossterms when dealing with multicomponent signals. | Bypasses the major drawback of the Short-Time Fourier Transform (STFT), which falsely assumes the signal is stationary. |
| **Wigner-Ville (WV) distribution** | Joint density spectrum displays very good localization properties; highly concentrated around instantaneous frequencies. | Highly susceptible to noise (very noisy output). |
| **Choi-Williams method** | Effectively reduces interference compared to Wigner-Ville. | Fails to satisfy all desired theoretical properties for a perfect time-frequency distribution. |
| **Artificial Neural Networks (ANN)** | Learns complex mappings to discover hidden patterns; performs real-time recognition; drastically curtails required subject training time. | — |
| **Fuzzy Logic** | Tolerates contradictory biomedical data; discovers hidden patterns; emulates human decision-making more closely than ANNs. | — |
| **Higher-order Statistics (HOS)** | Bispectrum (3rd order) suppresses Gaussian noise; carries phase and magnitude info to recover system impulse functions; detects non-Gaussianity. | Applied specifically when analyzing random, non-linear time series. |

---

## Reference 4 — Choi, 2023

H.-S. Choi, "Electromyogram (EMG) Signal Classification Based on Light-Weight Neural Network with FPGAs for Wearable Application," *Electronics*, 2023.

---

### 4.1 The edge vs. cloud paradigm

- **The server problem** — Sending raw biosignals to a cloud server drains heavy hardware resources, demands high transmission power, creates massive user-security vulnerabilities, and causes latency (slow response times).
- **The edge/FPGA solution** — Performing inference directly on an edge Field Programmable Gate Array (FPGA) guarantees fast response times (no transmission delays), drastically improves healthcare data privacy, and lowers system power consumption. 

---

### 4.2 FPGA optimization & HLS pipelines

- **Parallel operation** — FPGAs feature a distributed structure that allows for simultaneous parallel computation with very low memory access overhead.
- **HLS `#pragma` commands** — By utilizing specific High-Level Synthesis (HLS) grammar commands like `#pragma HLS PIPELINE`, the compiler forces parallel computation within the hardware loops, drastically reducing real-time inference latency.
- **The feature extraction rule** — Instead of relying purely on brute-force network power, robust feature extraction prepares the data better. This allows the system to achieve high accuracy using significantly fewer model parameters, directly aligning with edge constraints.

---

### 4.3 Model compression & quantization trade-offs

To fit modern networks onto wearable FPGAs, three compression strategies are required: 
1. **Algorithmic network compression** 
2. **Computation compression** (e.g., MobileNet, TPU systolic arrays)
3. **Weight sparsity pruning & bit quantization** (e.g., using `QKeras` for bit optimization).

**The `ap_fixed` Quantization Trade-off:**
When defining fractional bit lengths (quantization) in HLS via `ap_fixed<W, I>` (where W is total bits and I is integer bits), you must balance hardware area against accuracy loss.

| Data Type | Accuracy | Hardware Resource Usage | Engineering Verdict |
|---|---|---|---|
| `ap_fixed<24, 6>` | ~96% | High (100% baseline) | Extremely accurate, but consumes too many DSPs/LUTs. |
| `ap_fixed<22, 6>` | ~95% | **Optimal (80% baseline)** | **The Sweet Spot:** Saves 20% hardware resources for only a ~1% drop in accuracy. |
| `ap_fixed<20, 6>` | ~86% | Very Low | Unusable; aggressive bit truncation mathematically destroys the network's predictive capability. |

---

### 4.4 Current HLS limitations & future outlook

- **The HLS optimization gap** — While HLS speeds up hardware deployment, the automatic optimization between latency and hardware resource mapping is still currently insufficient compared to manual Verilog/VHDL coding. More efficient implementation techniques must be developed in the future.
- **Application goals** — Lightweight edge ML enables seamless Human-Computer Interaction (HCI), rapid disease diagnosis, and secure user biometric authentication—all running locally on battery-powered wearable modules.

---

## Reference 5 — Phinyomark et al., 2018

A. Phinyomark, R. N. Khushaba, and E. Scheme, "Feature Extraction and Selection for Myoelectric Control Based on Wearable EMG Sensors," *Sensors*, vol. 18, no. 5, p. 1615, 2018.

---

### 5.1 Wearable HCI constraints & pipeline dependency

- **Form factor goals** — For EMG interfaces to be viable in daily life, they must be completely non-invasive and discreet (e.g., integrated into watches, armbands, jewelry, or concealed beneath clothing).
- **The dependency on feature extraction** — A standard EMG pattern recognition pipeline consists of four parts: pre-processing, windowing, feature extraction, and classification. Because features project complex signals into lower-dimensional spaces, the ultimate success of the classifier relies *almost entirely* on selecting high-quality, representative features.

---

### 5.2 Established filtering & windowing parameters

*This study explicitly defines standard boundaries for prepping real-time EMG signals on embedded systems.*

| Parameter | Configuration | Engineering Purpose |
|---|---|---|
| **Notch Filter** | 50 Hz | Targets regional power-line interference. *(Note: Contradicts Reaz et al. [Ref. 3], further highlighting the filtering debate).* |
| **Band-pass Filter** | 20–500 Hz | Utilizes a 4th-order digital Butterworth FIR filter to eliminate motion artifacts (<20 Hz) and high-frequency random noise (>500 Hz). |
| **Data Windowing** | 250 ms | Compresses enough temporal information to generate a reliable feature without delaying user response. |
| **Window Increment** | 125 ms (50% overlap) | Ensures smooth, continuous data streams. This specific overlap ratio is proven suitable for real-time execution on embedded systems. |

---

### 5.3 The low-resolution sampling bottleneck

- **The Nyquist constraint** — Historically, robust feature extraction methods were developed for medical-grade sensors sampling at or above the Nyquist rate (usually ≥ 1000 Hz). 
- **The problem with consumer wearables** — The study investigates whether these legacy features hold up under modern, low-resolution sensor conditions.
- **The verdict** — A lower sampling rate fundamentally fails to preserve enough control information. It is insufficient for the accurate classification of complex movements (e.g., 6 to 7 different hand/finger motions) when relying on a standard 6–8 channel armband array.

---

## Reference 6 — Li et al., 2021

W. Li, P. Shi, and H. Yu, "Gesture Recognition Using Surface Electromyography and Deep Learning for Prostheses Hand: State-of-the-Art, Challenges, and Future," *Frontiers in Neuroscience*, vol. 15, p. 621885, Apr. 2021.

---

### 6.1 Deep learning vs. traditional ML

Because sEMG signals are highly dependent on the user's subjective intention and real-time interaction environment, relying purely on raw signal amplitude is fundamentally flawed. Pattern recognition relies on multi-dimensional feature extraction. 

| Approach | Signal State Handling | Strengths & Limitations |
|---|---|---|
| **Traditional ML** | Steady-state signals | Based on feature engineering. Struggles to effectively train on raw data that is inconsistent, noisy, abstract, or highly dimensional. |
| **Deep Learning (DL)** | Transient-state signals | Based on feature learning. Uses deep hierarchical architectures to automatically extract high-level feature information across hidden layers. Perfectly suited for the dynamic, transient nature of gestures performed in everyday life. |

---

### 6.2 Sensor acquisition paradigms

The density of the electrodes directly impacts how the data is processed and what algorithms are required.

| Acquisition Method | Characteristics & Trade-offs |
|---|---|
| **Sparse multi-channel sEMG** | **Pros:** Low hardware cost; minimal data transfer required.<br>**Cons:** Extremely sensitive to domain changes (sensor shifts, muscle variations). |
| **High-density sEMG (HD-sEMG)** | **Pros:** Captures exact spatial and temporal MUAP distribution via a 2D array; offers massive data quantity for highly accurate DL model inputs.<br>**Cons:** Demands highly complex analog front-ends and severe computational resources. |

---

### 6.3 DL preprocessing & input pipelines

Before classification can occur in a deep learning architecture, the signal undergoes three mandatory preprocessing steps: **smoothing**, **normalization**, and **segmentation**.

When configuring inputs for a DL network, there are two primary architectural strategies:
1. **Manual feature engineering:** Traditional features are extracted first to increase the mathematical data density before feeding it to the network. This is heavily required when using sparse multi-channel sEMG arrays.
2. **End-to-end learning:** The raw sEMG signal is fed directly into the deep neural network without handcrafted feature wrappers, allowing the architecture itself to learn the representations organically.

---

### 6.4 Network architecture suitability

| Architecture | Implementation Traits |
|---|---|
| **RNN** | Highly dominant for extracting patterns from temporal/sequential information, aligning perfectly with the time-series nature of sEMG. |
| **TCN** | Temporal Convolutional Networks outperform classical ML on embedded devices, easily managing strict power budgets and limited compute resources. |
| **ANN / FCNN** | Due to their structural simplicity, fully connected and artificial neural networks are commonly deployed for both offline and real-time identification. |
| **Unsupervised Learning** | Eliminates manual tagging requirements, but self-training strategies often spike classification errors when the data distribution mutates. |

---

### 6.5 Performance evaluation metrics

There is a distinct gap between evaluating algorithmic performance in a lab versus a real-world system. This study strongly emphasizes the necessity of real-time assessment over offline testing.

| Evaluation State | Standard Metrics Used |
|---|---|
| **Offline Performance** | Accuracy, recall, precision, and standard deviation. |
| **Real-Time Assessment** | Overshoot, throughput, path efficiency, completion rate, average speed, and stopping distance. |

---

### 6.6 Variability & evaluation protocols

sEMG is a heavily non-stationary physiological signal. Its statistical characteristics drift over time due to individual physical differences and environmental factors. This high variability causes "domain shifts" (the training data and real-world test data have different distributions). 

To properly assess model robustness, specific evaluation protocols are required:
* **Intra-session:** The model is trained and evaluated using data from different trials within the exact same session (the electrodes are never removed or shifted).
* **Inter-session:** The model is evaluated on data from the same subject, but across completely different sessions (meaning the electrodes were removed and later reattached, introducing spatial shifts).
* **Intra-subject:** The model is trained and evaluated solely on the variance of one specific user.

---

### 6.7 Deployment constraints & future outlook

- **User constraints** — For prosthetics, system real-time capability and physical compactness are the two ultimate deciders of user satisfaction. These demands put massive pressure on the data acquisition, storage, and embedded processing pipelines.
- **Clinical translation** — Models should not just be assessed offline. They must be evaluated in daily life scenarios with amputees, medical staff, and researchers utilizing specialized simulation scales.
- **Telecommunications synergy** — The integration of 5G and IoT offers a highly viable solution to current bottlenecks. 5G vastly accelerates information transfer speeds between wearable nodes, while IoT architectures drastically improve the large-scale collection and sharing of complex biomedical data.
- **Current bottlenecks** — Progress in DL-based pattern recognition is currently throttled by four primary issues: extreme signal variability, a lack of standardized training data, strict hardware resource limitations, and a severe lack of structured clinical evaluation conditions.

---

## Reference 7 — Lima et al., 2024

G. M. Lima, D. P. Campos, and R. G. Mantovani, "A Review on the Recent use of Machine Learning for Gesture Recognition using Myoelectric Signals," *ENIAC*, 2024.

---

### 7.1 Paradigms & algorithmic pipeline

* **The Non-Invasive Foundation** — Surface Electromyography (sEMG) serves as a vital non-invasive approach to capture muscle bio-potentials and track muscle activity over time.
* **The Open Architecture Debate** — System designers generally divide pipelines between Classical ML and Deep Learning (DL). Currently, there is no universal consensus or standard method for optimized feature extraction or classification.
* **Learning Paradigm Superiority** — In myoelectric gesture systems, **supervised learning** remains far superior to other training modalities, significantly outperforming reinforcement learning approaches in both adaptability and classification accuracy.

| Pipeline Stage | Operational Focus |
|---|---|
| **1. Data Acquisition** | Raw capture of muscle bio-potentials via surface electrode nodes. |
| **2. Data Preparation** | Signal conditioning through filtering, amplification, preprocessing, and feature extraction. |
| **3. Classification** | Passing processed feature tokens into a trained predictive model to determine intent. |
| **4. Model Evaluation** | Testing the safety, accuracy, and operational bounds of the system. |

---

### 7.2 Four generations of feature extraction

Unlocking hidden patterns within complex biomedical signals requires choosing the right mathematical lens. The literature frames the evolution of feature extraction techniques into four clear generations:

| Generation | Domain Type | Primary Engineering Focus |
|:---:|---|---|
| **1st** | Time Domain (TD) | Lowest computational footprint; tracks structural signal amplitude changes over time. |
| **2nd** | Frequency Domain (FD) | Maps spectral energy distributions; highly reliable for tracking structural muscle fatigue. |
| **3rd** | Joint Time-Frequency Domain (TFD) | Localizes transient, non-stationary muscle behaviors without dropping time orientation. |
| **4th** | Signal Decomposition & Sparse Domains | Extracts highly abstract, low-density underlying structures from compound multi-sensor biosignals. |

---

### 7.3 Hybrid synergy & gesture scale limits

* **The Power of Combined Architecture** — Relying exclusively on one feature domain creates a ceiling for accuracy. Experimental data demonstrates that a **combined approach**—hybridizing traditional time-spectral analysis with automatic CNN-based feature learning layers—consistently outperforms single-domain pipelines.
* **The Gesture Complexity Wall** — The predictive capacity of an embedded classifier degrades sharply as the physical demands of the application grow:

> ⚠️ **The Class Scaling Bottleneck:**
> The vast majority of modern studies restrict evaluation to **fewer than 10 gestures**. Attempting to scale vocabulary beyond this boundary increases the number of target classes, leading to severe overlap in feature distributions and significantly lower distinction accuracy.

---

### 7.4 Evaluation gaps & dataset imbalances

* **The Accuracy Trap** — A massive blind spot in contemporary myoelectric research is the overwhelming reliance on *accuracy* as the solitary validation metric.
* **The Imbalance Mitigation Rule** — Because real-world sEMG data streams are heavily imbalanced (characterized by extensive periods of muscle rest interspersed with brief, transient bursts of gesture activity), raw accuracy fails to catch systemic false positives.
* **Design Choice** — Embedded and clinical system developers must look past baseline accuracy and actively integrate **Recall** and **F1-Score** to accurately measure real-world performance on uneven datasets.

---

## Reference 8 — Peres, 2016

L. B. Peres, "Classificação de atividade eletromiografia facial de indivíduos saudáveis e com hanseníase por meio de máquina de vetores de suporte," *Dissertação (Mestrado em Ciências)* - Universidade Federal de Uberlândia, Uberlândia, 2016.

---

### 8.1 Physiological nature & signal generation

* **Bioelectric Origin** — The electromyographic (EMG) signal originates directly from the electrical activity across the excitable membranes of muscle cells.
* **Stochastic Architecture** — The signal is represented as an electrical voltage traveling over time. It is a stochastic summation of all concurrent signals within the muscle volume, heavily affected by anatomy, physiological variations, hardware front-ends, and environmental capture noise.
* **Motor Control Mechanics** — Muscle fibers are innervated in functional groups called Motor Units (MUs). Activation generates a Motor Unit Action Potential (MUAP). Repetitive firings form a Motor Unit Action Potential Train (MUAPT). Heightened voluntary contraction force directly expands the density and physical intensity of these trains.
* **Activity-Driven Signaling Paradigm** — The recorded sEMG profile is fundamentally an expression of the specific *functional activity performed* rather than a static signature of the muscle tissue itself. Consequently, a single target facial muscle can yield entirely different signal characteristics and features depending on the physical movement executed.

---

### 8.2 Conditioning & the oversampling dilemma

* **Front-End Pre-Amplification** — Facial sEMG potential variations manifest at nominal amplitudes on the order of microvolts ($\mu	ext{V}$), demanding high-gain pre-amplification right at the sensor stage.
* **Filtering Topologies** — Conditioning blocks can be designed as active (op-amp circuits), passive (resistor, capacitor, inductor networks), or discrete digital processing implementations.

| Parameter | Configuration | Engineering Target |
|---|---|---|
| **High-pass Filter** | 20 Hz | Attenuates low-frequency baseline fluctuations and slow motion artifacts. |
| **Low-pass Filter** | 500 Hz | Suppresses high-frequency noise outside the primary physiological sEMG band. |

> ⚠️ **The Nyquist Oversampling Paradox in Practice:**
> While standard Nyquist math states a signal can be perfectly reconstructed if sampled at twice its highest frequency ($2 	imes 500	ext{ Hz} = 1000	ext{ Hz}$), operating exactly at this baseline threshold causes severe data truncation and a critical loss of transient biomedical information.
> *System Choice:* The architecture implements a **5000 Hz sampling rate** (5x Nyquist), ensuring high temporal resolution and data preservation for the microcontroller.

---

### 8.3 Feature preparation, normalization & reduction

* **Divide-and-Conquer Windowing** — Sustained 20-second dynamic facial contractions are fragmented into concise, localized **5-second static windows** to track metrics individually and map features into a multidimensional storage table.
* **Multi-Domain Feature Mapping** — Due to the unknown a priori significance of features for clinical leprosy diagnostics, a broad extraction matrix is applied across three distinct wave representations:

| Target Signal Source | Extracted Domain Matrix | Engineering Rationale |
|---|---|---|
| **Filtered Signal** | Amplitude, Frequency, Randomness | Basal mapping of macro muscle energy profiles and stochastic dynamics. |
| **Instantaneous Amplitude** | Signal Envelope | Computed via the **Hilbert Transform** to extract the true magnitude envelope of the contraction. |
| **Instantaneous Frequency** | Amplitude, Randomness | Tracks non-stationary properties (where frequency shifts over time) to isolate dynamic spectral peak shifts. |

* **Z-Score Normalization** — To ensure scale independence and prevent high-amplitude features from disproportionately biasing the optimization space, all data matrices undergo uniform Z-score standardization.
* **PCA-Based Quality Control & Reduction** — Principal Component Analysis (PCA) is applied to project the high-dimensional matrix into a lower-dimensional representation. 

> ⚠️ **The Data Verification Sieve:**
> To guarantee that physical data collection was performed flawlessly, the first two principal components (PC1 and PC2) are cross-plotted as a scatter diagram colored by activity type. If the data collection is compromised or noisy, these clusters fail to isolate, signaling that the dataset is fundamentally corrupted and invalidating any subsequent classification analysis.

* **Feature Vector Compaction ("Less is More")** — Testing demonstrates that feeding *all* extracted features simultaneously into the classifier degrades the performance and "hit rate" drastically. Because the precise structural mechanisms by which leprosy affects muscle contraction are physiologically unknown, brute-force input mapping introduces massive informational redundancy. Compacting feature strings via automated parameter loops is mathematically required.

---

### 8.4 SVM classification & performance metrics

* **Hyperplane Separation** — The Support Vector Machine (SVM) algorithm maps multidimensional inputs to establish an optimal dividing hyperplane, maximizing the physical boundary margin between two target classes.
* **Kernel Transformations & Structural Separability** — Because the multidimensional biological data cannot be neatly split or plotted in a basic standard plane, non-linear kernel transformations are applied to project signals into higher-dimensional boundary spaces. 
* **Kernel Sensitivity & Muscle Specificity** — Experimental data confirms that each facial muscle provides drastically varied classification outputs depending on the chosen SVM kernel function type (e.g., Linear vs. RBF). Selecting a muscle-tailored kernel configuration changes the model's accuracy bounds significantly.
* **Cross-Validation Rigor** — Employs a **10-Fold Cross-Validation** structure ($k = 10$). The full feature pool is split into 10 partitions: 90% is consumed for hyperplane training, while the remaining 10% is used to validate model robustness.
* **Statistical Performance** — System validation tracks True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) via a 2x2 matrix to yield Accuracy, Sensitivity, Specificity, and Precision.

> ⚠️ **The Isolated Precision Trap:**
> An isolated high precision score does not prove that a classifier is clinically efficient; it merely indicates how uniformly stable and reproducible a specific prediction was. Developers must cross-verify precision with **sensitivity** to guarantee that the system successfully identifies true pathological conditions without missing patients.

---

## Reference 9 — Pérez-Reynoso et al., 2022

F. Pérez-Reynoso, N. Farrera-Vazquez, C. Capetillo, N. Méndez-Lozano, C. González-Gutiérrez, and E. López-Neri, "Pattern Recognition of EMG Signals by Machine Learning for the Control of a Manipulator Robot," *Sensors*, vol. 22, no. 9, p. 3424, 2022.

---

### 9.1 Physiological mechanics & simultaneous control limits

* **Peripheral Nerve Exploration** — Electromyography (EMG) is physiologically defined as the electrical exploration of peripheral nerves through the stimulation of muscles to capture the resulting action potentials.
* **Contraction Modalities** — Muscle contractions are categorized into two fundamental operational states:
  * *Isometric Contraction:* A static form of muscle engagement where force is produced without any appreciable change in muscle length.
  * *Isotonic Contraction:* A dynamic form of muscle engagement where the muscle shortens or lengthens while the force of contraction remains relatively constant.
* **The Fatigue & Motion Barrier** — To prevent unwanted motion artifacts from involuntary tremors and protect against muscle fatigue during biceps brachii capture, recording sessions are kept brief—strictly limited to a maximum duration of **45 seconds**.
* **The Simultaneous Degree-of-Freedom (DoF) Wall** — Traditional pattern recognition architectures suffer from sequential and binary control constraints, limiting user control to a single prosthetic hand function at a time. Designing continuous signal models is required to enable natural hand movements consisting of simultaneous activations across multiple DoFs.

---

### 9.2 Conditioning & hardware front-end topologies

* **Basal DC Correction** — Raw signals are processed through a basal correction circuit to eliminate DC offset voltages caused by involuntary tremors, user shifting, or poor electrode-skin contact.
* **Analog High-Pass Clipping Prevention** — Active high-pass filters are implemented right at the hardware stage to strip away the DC bias voltage, protecting the high-gain operational amplifiers (op-amps) from clipping or reaching their maximum power limits.
* **Active Bandpass Topology** — To capture the necessary biological bandwidth of the EMG signal (0.5 Hz to 5 kHz), the hardware implements a second-order active Butterworth bandpass filter with unity gain.
  * *Hardware Spec:* Delivers a roll-off of 40 dB per decade using high-impedance **TL084 operational amplifiers**, precision resistors, and electrolytic capacitors.
* **Spectral Verification** — The real-time implementation of a Fast Fourier Transform (FFT) is executed to inspect the frequency spectrum, ensuring precise identification of cutoff frequencies before configuring analog or digital filters.

---

### 9.3 Signal preprocessing, normalization & digital filtering

* **First-Order Digital Smoothing** — Downstream processing implements a first-order digital low-pass filter. This mathematically lightweight design minimizes processing latency during real-time hardware execution on embedded controllers.
* **Unit-Variance Normalization** — To handle varying voltage thresholds across separate channels and trials, the raw acquisition vector $p$ is normalized by subtracting the mean and scaling the standard deviation to 1 ($\mu = 0, \sigma = 1$). This step significantly reduces the backpropagation computational learning cost of the network.

---

### 9.4 MNN classification & state machine translation

* **Four-Class Contraction Taxonomy** — A Multilayer Neural Network (MNN) is trained to identify four distinct temporal contraction waveforms:
  1. *Sharp Muscle Pulse (SMP):* An instantaneous 1-second contraction followed by a 5-second relaxation window.
  2. *Smooth Muscle Pulse 3 s (SMP3):* A 3-second contraction followed by a 5-second relaxation window.
  3. *Smooth Muscle Pulse 5 s (SMP5):* A 5-second contraction followed by a 5-second relaxation window.
  4. *Noise Involuntary Movements (NIM):* Captures resting baselines and unexpected tremors (mapped as a "total stop" state).
* **One-Hot Encoding Integration** — Waveforms are mapped to distinct categorical integers ($1$ to $4$) through supervised learning labels.
* **Ultra-Low Cost Decision Logic** — To keep the firmware loop computationally cheap on embedded MCUs, the model utilizes simple `if-else` triggers and direct returns:
  $$	ext{If Network Output } = 	ext{Integer } [1	ext{--}4] \implies 	ext{Enable Pin } (1), 	ext{ else } (0)$$
* **State Machine Activation** — These discrete digital pulses act as transition signals for a coordinate state machine, translating classified wave states into Cartesian movements $(x, y, z)$ on a physical 3-DOF robotic manipulator.

---

## Reference 10 — Fathi et al., 2026

T. Fathi, M. A. M. Abdullah, and B. Shukr, "Machine and Deep Learning Model for EMG Signal Classification: A New Performance-Cost Analysis Across CPU and GPU Architectures," *The International Arab Journal of Information Technology*, vol. 23, n. 2, p. 315–324, mar. 2026.

---

### 10.1 sEMG sensor limitations, signal noise & cost selection factors

* **The Impedance & Cross-Talk Barrier** — A primary disadvantage of surface electromyography (sEMG) sensors is reduced classification accuracy stemming from high skin impedance and signal cross-talk leaking from adjacent muscle groups.
* **Inherent Complexity Parameters** — sEMG wave fields are highly chaotic and noisy, heavily influenced by physical factors such as target electrode placement variations and individual muscle strength fluctuations.
* **Traditional Classifier Failures** — Conventional machine learning algorithms are exceptionally sensitive to noise and struggle to model or capture highly abstract, complex structural patterns.
* **The Cost-Selection Principle** — Choosing the optimal algorithm for an EMG deployment cannot rely on accuracy alone; it requires a performance-to-cost analysis where **cost** is directly tied to total execution time and specific hardware architecture requirements.

---

### 10.2 Spatial-temporal network fusion & parallel processing overhead

* **Parallel Computing Acceleration** — Deep learning models for bio-signal analysis exhibit a high degree of mathematical parallelizability, underscoring the efficiency of deploying parallel processing structures (GPUs) over sequential alternatives (CPUs).
* **Hybrid CNN-LSTM Decomposition** — Combining Convolutional Neural Networks with Long Short-Term Memory networks resolves single-domain limits by explicitly splitting feature parsing:
  * *Spatial Extractor (CNN):* Identifies localized spatial topologies across multiple parallel EMG channels and learns short-term, time-based dependencies within brief data windows.
  * *Temporal Modeler (LSTM):* Processes the resulting spatial representations to map long-term sequential temporal dynamics over extended intervals.
* **The Parametric Deep Learning Tax** — Deep architectures (LSTM and CNN-LSTM) require significantly extended training durations compared to traditional ML. This training lag is driven by massive parameter scaling and a higher number of sequential computational steps required per parameter during execution.

---

### 10.3 Classifier benchmarking & application-dependent logic

* **The Validation Matrix** — A standard Confusion Matrix is integrated into the validation pipeline as a tabular paradigm to directly cross-reference and map ground-truth class labels against model-predicted categories.
* **Random Forest (RF) Performance vs. Cost** — The Random Forest algorithm yields the peak classification accuracy at **98.16%**, but introduces a heavy structural cost by consuming the absolute maximum number of training nodes.
* **Decision Tree (DT) Efficiency** — The Decision Tree model serves as the most efficient lightweight alternative; it is structurally simpler, demands the shortest training and testing times, minimizes training node overhead, and still maintains a strong accuracy profile of **95.46%**.
* **The Application Dependency Rule** — There is no universally superior classifier; selecting the ideal model depends entirely on the design constraints of the specific target EMG application (e.g., lightweight embedded microcontrollers vs. high-throughput cloud diagnostics).

---

## Reference 11 — Souza, 2023

W. M. de Souza, "Classificador Random Forest para eletromiografia de superfície: uma abordagem em FPGA," Master's thesis, Dep. Engenharia Elétrica, Universidade Federal do Rio Grande do Sul, Porto Alegre, 2023.

---

### 11.1 Embedded design objectives, biological filtering & hardware front-end

* **Untethered Edge Execution** — The primary engineering objective is the development of a low-cost, small-scale embedded system capable of local signal acquisition, processing, and pattern recognition without requiring specialized laboratory infrastructure.
* **The Biological Filter Paradigm** — In surface readings (sEMG), physical distance separates the electrode from the target muscle unit. Biological tissue functions as a natural low-pass filter, attenuating frequencies above 400 Hz.
* **Spectral and Amplitude Bounds** — Relevant sEMG physiological information is bounded between 10 Hz and 500 Hz (with the bulk of useful energy concentrated between 20 Hz and 150 Hz). Unconditioned amplitudes can swing up to $\pm 5000\ \mu	ext{V}$ in athletic subjects.
* **Hardware Acquisition Front-End** — The signal acquisition stage implements an instrumentation amplifier with a high gain of **300**, followed by a two-stage active filtering circuit (high-pass and low-pass). The 8 parallel channels are digitized via a 16-bit Delta-Sigma ($\Delta\Sigma$) ADC per channel.
* **Target Domain Vulnerabilities** — While time-domain analysis allows raw signals to be processed directly without computationally intensive mathematical transformations, it remains exceptionally vulnerable to extrinsic noise sources (such as power grid line interference and electrode-skin contact friction).

---

### 11.2 Bipolar electrode protocols & sliding window segmentation

* **Bipolar Electrode Alignment** — Signal acquisition utilizes a differential bipolar topology (two differential electrodes + one reference electrode). Differential nodes are oriented strictly along the muscle fiber direction, deliberately avoiding marginal muscle borders due to low Motor Unit Action Potential (MUAP) density in those regions.
* **Sliding Window Segmentation** — To capture information from stochastic time-series signals, raw data streams are segmented into sliding windows of **maximum 300 ms** with a **50% window overlap**.

---

### 11.3 Feature selection, tree ensembles & Random Forest mechanics

* **Feature Selection Primacy** — Selecting the correct feature set is mathematically more critical to control system performance than the choice of classification technique itself. Classifier accuracy remains directly tied to feature quality.
* **Supervised vs. Unsupervised Taxonomy** — Machine learning methods are divided by architectural initiation: **supervised** pipelines rely on predefined input-target pairs, whereas **unsupervised** models cluster data autonomously using quality metrics without ground-truth targets.
* **Decision Trees & Random Forests** — Decision trees offer structural transparency, balancing accuracy through node counts, leaf counts, tree depth, and split attributes. Random Forests expand this parametric balance via bootstrapping to handle large-scale datasets efficiently.

---

### 11.4 DSP-less FPGA VHDL design & mathematical workarounds

* **FPGA Selection Constraints** — Device selection depends heavily on hardware fabric limits: available Look-Up Tables (LUTs), Flip-Flops (FFs), and total available I/O pin counts.
* **DSP-Less Math Implementations** — Executing multiplication and division on FPGAs lacking dedicated DSP slices is extremely hardware-costly. Custom VHDL algorithms bypass this hardware barrier:
  * *Multiplication:* Implemented via a **successive addition algorithm**.
  * *Division:* Implemented via a **successive subtraction algorithm**.
  * *Recursive Summation:* Implemented using physical hardware accumulators linked directly to a dedicated RAM block to manage recursive time-domain summations.

---

### 11.5 Scaling dynamics: MCU vs. FPGA vs. System-on-Chip (SoC)

* **The Microprocessor Channel Bottleneck** — While cheap microprocessors (MCUs) are viable for small single-channel setups, increasing the channel count causes microprocessors to suffer a severe, steep performance degradation compared to FPGA fabrics.
* **FPGA Parallelism Supremacy** — Direct execution on an FPGA fabric achieves a drastically superior accuracy-to-power consumption ratio over multi-core CPUs during multi-channel feature extraction.
* **System-on-Chip (SoC) Hybrid Architecture ("The Gold Standard")** — Integrates a microprocessor core alongside FPGA programmable logic inside a single chip package:
  * **FPGA Logic Fabric:** Handles raw hardware-intensive math operations (filtering, recursive summation, matrix operations).
  * **Microprocessor Core:** Manages sequential control tasks, protocol communications, and system management.

---

## Reference 12 — Xavier, 2021

R. T. Xavier, "Classificação com Deep Learning de Sinais de uma Interface Neural HDsEMG para Acionamento de Neuropróteses Transradiais," Doctoral thesis, Faculdade de Engenharia, Universidade Estadual Paulista "Júlio de Mesquita Filho", Ilha Solteira, 2021.

---

### 12.1 High-dimensional Deep Learning bottlenecks & CNN mechanics

* **Statistical Generalization Barrier** — High-dimensional distributions introduce an exponential growth in possible state configurations, which quickly exceeds available training samples and creates a statistical generalization bottleneck.
* **Computational Complexity** — Algorithms handling high-dimensional distributions face intractable calculations scaling exponentially with input dimensions (citing LeCun, Bengio, & Hinton, 2015).
* **CNN Kernel Operations** — Convolution is a linear matrix multiplication operation between two functions that generates a feature map across receptive fields, incrementally shifted by a *stride* parameter (typically 1).

---

### 12.2 HDsEMG acquisition, filtering & experimental protocol

* **Sampling & Bandpass Preprocessing** — 64-channel High-Density sEMG (HDsEMG) streams are acquired at **2000 Hz** and conditioned using a **10 Hz 4th-order high-pass Butterworth filter** to eliminate low-frequency drift and motion artifacts.
* **Kinematic Data Isolation** — Kinematic recordings served strictly as an external validation benchmark for HDsEMG signals and were excluded from ML training/classification inputs.
* **Subject Cohort & Reference Setup** — Trials were conducted with 9 healthy adult volunteers (male/female, aged 18–30), using reference electrodes positioned near the elbow joint.

---

### 12.3 Forearm kinematic taxonomy & 9-class gesture matrix

* **85% Forearm Contraction Mapping** — The trial protocol defined 9 distinct gesture classes capturing roughly **85% of total forearm superficial muscular contraction**:
  1. *Wrist Flexion (MFP)*
  2. *Wrist Hyperextension (MHP)*
  3. *Radial Deviation (MDR)*
  4. *Ulnar Deviation (MDU)*
  5. *Clockwise Wrist Circumduction (MCH)*
  6. *Counter-Clockwise Wrist Circumduction (MCA)*
  7. *Interphalangeal & Metacarpophalangeal Flexion w/o Thumb (MFI)*
  8. *Hand Closure / Fist (MFM)*
  9. *Rest / Forearm Inertia (RPO)*
* **Dataset Volume** — 5 series of 5 repetitions per gesture were recorded, yielding **25 stored recordings per class** across all 64 HDsEMG channels.

---

### 12.4 End-to-end DL paradigms & parallel compute scaling

* **Multimodal Direct Learning** — Deep Learning (DL) models learn classification tasks directly from raw input signals across diverse data modalities, including images, text, and audio.
* **Automated Feature Extraction vs. Traditional ML** — Traditional ML workflows depend on manual feature extraction prior to training. DL acts as a specialized subfield of ML that performs end-to-end learning, capturing raw input data alongside target tasks to make feature learning fully automatic.
* **GPU Parallel Acceleration & Cloud Clusters** — While DL delivers high classification performance, it requires substantial computational power. Leveraging high-performance GPUs with parallel architectures, cloud computing, or GPU clusters significantly reduces model training latency from days down to hours or minutes.

---

### 12.5 Imbalance evaluation metrics & real-time neuroprosthetic constraints

* **The Class Imbalance Accuracy Fallacy** — Evaluating models using raw classification accuracy as the sole performance metric yields deceptive or misleading conclusions when training datasets contain imbalanced classes.
* **Confusion Matrix Diagnostics** — Imbalanced datasets require structural evaluation metrics derived from confusion matrices (citing Bradley, 1997) to properly track algorithmic behavior across correct and incorrect classifications.
* **Compute Costs in Neuroprosthetic Deployment** — Model training and processing latency scale directly with network parameter size. Higher computational cost may be acceptable if a neuroprosthetic model classifies inputs only once. However, neuroprosthetic systems requiring continuous, real-time online learning demand strict compute minimization to maintain acceptable latency bounds.

---

## Reference 13 — Guo et al., 2024

Y. Guo et al., "FPGA-based Lightweight QDS-CNN System for sEMG Gesture and Force Level Recognition," *IEEE Transactions on Biomedical Circuits and Systems*, 2024.

---

### 13.1 ML vs. DL paradigms & FPGA edge acceleration

* **Hand-Crafted Feature Bottleneck** — Traditional ML classifiers (SVM, Decision Trees) rely on manual extraction of time and frequency domain features (mean, variance, PSD). This yields high model complexity, excessive latency, and heavy hardware resource consumption.
* **Automated DL Feature Learning** — Deep Learning automatically extracts features, eliminating manual feature engineering overhead. DL architectures can be further fine-tuned using quantization and pruning to enable deployment on resource-constrained edge hardware.
* **FPGA Parallel Computing Acceleration** — FPGAs accelerate DL model processing through parallel computing and optimized hardware architectures, achieving rapid inference speeds and minimal latency.

---

### 13.2 Signal acquisition, zero-phase filtering & STFT spectrograms

* **Subject Cohort & Acquisition Setup** — Trials involved 22 healthy participants (14 males, 8 females, average age 28, range 20–51). Signals were acquired across 8 monopolar sEMG channels at a **1 kHz** sampling rate using a custom device while subjects executed 18 right-hand gestures paired with force levels.
* **Zero-Phase Conditioning Pipeline** — Preprocessing utilized an **8th-order Butterworth bandpass zero-phase filter (20–250 Hz)** to eliminate noise without phase distortion (preserving sEMG waveform shape and timing), followed by a **50 Hz notch filter** to eliminate powerline interference.
* **Windowing & STFT Feature Mapping** — Signals were framed using a **128 ms window** with a **64 ms step size** (50% overlap). Short-Time Fourier Transform (STFT) converted multi-channel sEMG into 2D spectrogram maps, capturing local features and non-linear properties better than raw signal inputs.

---

### 13.3 Network architecture, quantization & validation setup

* **UL-DSC & CA-GAP Architecture** — The network combines Ultra-Lightweight Depthwise Separable Convolution (UL-DSC) with Channel Attention-Global Average Pooling (CA-GAP). UL-DSC modifies activation functions and layer dimensionality to minimize parameter counts and arithmetic operations.
* **Parallel Kernel Execution** — Designed a custom depthwise convolution strategy capable of processing 9 data points within a $3 	imes 3$ convolution kernel in a single clock cycle at full hardware expansion.
* **8-Bit Fixed-Point Quantization** — Evaluated quantization approaches and bit-widths. 8-bit fixed-point quantization reduced model size by **4×** with only a **0.28% accuracy loss**, striking an optimal balance between model reduction and hardware implementation simplicity compared to dynamic range quantization.
* **Validation Strategy** — Merged dataset samples were randomly shuffled: 80% was allocated to training and validation via 10-fold cross-validation, while 20% was reserved as a holdout test set.

---

### 13.4 Hardware implementation & performance metrics

* **Development & Hardware Setup** — Model design and training were executed in Python 3.9 and TensorFlow on an Intel Core i7-12700 CPU with an RTX 3060 GPU. The accelerator was synthesized in Vivado 2020.1 using RTL coding and deployed on a Xilinx ZCU102 evaluation board (Zynq XCZU9EG-2FFVB1156 SoC running PYNQ, equipped with 600k Logic Cells, 32.1MB Memory, and 2520 DSP Slices).
* **Ultralight Footprint** — The finalized model contains only **5.0 k parameters** and a total size of **0.026 MB**.
* **Classification Accuracy** — Achieved an average accuracy of **94.92%** across all tasks (Static gesture recognition: **97.27%**, Force-level gesture recognition: **89.76%**).
* **Hardware Performance Profile** — Operating in 8-bit precision, the FPGA accelerator achieved a single-frame inference time of **41.9 µs**, throughput of **78.6 GOP/s**, and power consumption of **0.317 W**.

---

## Reference 14 — Tam et al., 2020

S. Tam, M. Boukadoum, A. Campeau-Lecours, and B. Gosselin, "A Fully Embedded Adaptive Real-Time Hand Gesture Classifier Leveraging HD-sEMG and Deep Learning," *IEEE Transactions on Biomedical Circuits and Systems*, vol. 14, no. 2, pp. 232–243, Apr. 2020.

---

### 14.1 HD-sEMG muscle activation maps & instantaneous latency reduction

* **Image-Based Spatial Paradigms** — Leverages High-Density sEMG (HD-sEMG) arrays on the forearm to record multi-muscle activity patterns, transforming 2D spatial voltage signals into "muscle activation maps" (heat maps of forearm muscular activity).
* **Instantaneous Latency Elimination** — Traditional feature-based models (TD/FD) require collecting an extended time window to compute mathematical features before classifying. In contrast, image-based CNN inference evaluates instantaneous spatial frames, eliminating windowing latency and enabling near-instantaneous control loops.
* **Wearable Usability Objectives** — Combines HD-sEMG with deep learning to simplify physical electrode alignment, enabling convenient, intuitive, low-latency wearable devices suited for daily usage.

---

### 14.2 Stochastic signal characteristics, shift invariance & normalization

* **Zero-Mean Gaussian Instability** — Surface EMG signals model as a zero-mean Gaussian process. Consecutive instantaneous frames exhibit poor correlation and significant variance within the same gesture, introducing unstable frame-to-frame predictions.
* **Normalization & Transfer Learning** — Applying Batch Normalization at the input accelerates network training, mitigates contraction strength variance, and preserves relative spatial muscle activation maps. This normalization enables transfer learning across subjects who exhibit varying muscle strengths but possess similar underlying anatomical structures and limb dimensions.
* **Spatial Shift Invariance** — Employing a shift-invariant Convolutional Neural Network (CNN) architecture reduces sensitivity to precise physical electrode positioning and alignment errors.
* **Feature Engineering Bypass** — Eliminates hand-crafted feature engineering dependencies required by classical classifiers (LDA, SVM), allowing the CNN to automatically extract spatio-temporal muscle patterns.

---

### 14.3 Network architecture & hyperparameter profile

* **Framework & Layer Composition** — Designed and trained using PyTorch. The lightweight CNN applies:
  * **Input Layer:** Batch Normalization directly at the network input
  * **Hidden Layers:** Batch Normalization applied after every Convolutional layer, paired with **Leaky ReLU** activation functions ($	ext{negative slope} = 0.01$)
* **Hyperparameter Matrix** — Trained for **12 epochs** using the **Adam optimizer** with a learning rate of **0.01** and a **cross-entropy loss function**.
* **Embedded Optimization Constraints** — Network footprint is lightweight to balance computational resource constraints and real-time execution bounds on embedded hardware target platforms.

---

### 14.4 Offline vs. real-time reliability & sliding majority voting

> ⚠️ **The Real-Time Translation Gap:**
> High offline classification test accuracy translates poorly to real-time reliability for human users. Untrained subjects struggle to achieve intuitive control without continuous closed-loop feedback, highlighting that real-time evaluation metrics are required to judge system performance over static offline benchmarks.

* **Sliding Majority Voting Post-Processing** — Implements a sliding multi-vote majority filter to resolve instantaneous frame-to-frame variance. At each sampling/inference iteration, the newest single inference vote is pushed into the voting buffer while the oldest vote is dropped (`FIFO` structure).
* **Post-Processing Trade-Offs** — Increasing the majority voting window size improves static gesture recognition stability at negligible computational cost, though classification errors persist during rapid dynamic gesture state transitions.

---

## Reference 15 — Lu et al., 2023

J. Lu et al., "EffiE: Efficient Convolutional Neural Network for Real-Time EMG Pattern Recognition System on Edge Devices," in *2023 11th International IEEE/EMBS Conference on Neural Engineering (NER)*, 2023.

---

### 15.1 Deep learning on edge devices & transfer learning strategy

| Concept | Description |
|---|---|
| **GPU Dependency vs. Edge Reality** | Deep Learning (DL) eliminates manual feature engineering by automatically learning spatial representations from 1D/2D sensor arrays, but standard DL inference requires high-performance GPUs. |
| **Edge Adaptation via Transfer Learning** | Mitigates computational barriers on edge hardware by pre-training a general source CNN on a large public dataset and fine-tuning it directly on user-specific sEMG data locally on the target device. |

---

### 15.2 Hardware acquisition & 2D-CNN network architecture

* **Acquisition Front-End** — Captures sEMG streams using an 8-channel Myo Armband sampled at 200 Hz (yielding 15 windows of 8 channels with 32 samples per window), formatted into 2D spatial sEMG images.
* **2D-CNN Layer Layout**:
  * *Layer 1:* Conv2D (48 filters, 3×3 kernel) ➔ PReLU activation ➔ Batch Normalization ➔ Spatial 2D Dropout (rate = 0.5) ➔ Max Pooling (1×2).
  * *Layer 2:* Conv2D (96 filters, 3×3 kernel) ➔ PReLU activation ➔ Batch Normalization ➔ Spatial 2D Dropout (rate = 0.5) ➔ Max Pooling (1×2).
  * *Output Layer:* Fully Connected (Dense) layer mapping extracted feature maps to multi-gesture target classes.

---

### 15.3 TFLite 8-Bit quantization & embedded deployment

* **Target Hardware Platform** — Executed locally on the **Sony Spresense** edge platform.
* **Post-Training Quantization** — Converts floating-point weights and inputs into 8-bit integer data types using TensorFlow Lite.
* **Footprint Optimization** — Shrinks the model size by nearly tenfold, dropping from **772 KB** (floating-point baseline) down to **73 KB**.

---

### 15.4 Offline training, edge fine-tuning & real-time trade-offs

| Optimization Phase | Hyperparameters & Setup | Operational Target |
|---|---|---|
| **Offline Source Model** | Dataset: NinaPro DB5<br>Optimizer: Adam (lr = 0.2, step decay = 0.9 per 1.5 epochs)<br>Epochs: 200<br>Batch Size: 384 | Learns generalized spatial sEMG feature representations across multi-subject datasets. |
| **On-Device Target Model** | Epochs: 10<br>Batch Allocation: 2 batches of sEMG data per feedforward pass<br>Final Footprint: Quantized to 73 KB | Rapidly adapts pre-trained weights to user-specific sEMG signals directly on edge hardware. |

* **Sliding Window & Overlap Dynamics** — Overlapping sEMG windows reduces subsequent image retrieval delays and continuous data transmission latency. However, step sizes must be balanced: excessively small step sizes degrade real-time responsiveness by increasing the number of prediction cycles required for the CNN model to transition to a newly performed gesture.
* **Real-Time Performance Profile**:
  * *Processing Latency:* $\le 160	ext{ ms}$ real-time end-to-end processing delay.
  * *Classification Accuracy:* **85%** real-time accuracy achieved directly on the target edge device post fine-tuning and 8-bit quantization.

---

## Synthesis — From Literature to FPGA Implementation Strategy

*This section consolidates the fifteen references above into an applied decision framework for a GERAM-style project: investigating efficient FPGA implementation of ML/DL algorithms for EMG signal processing, using the Molinari & Elias HD-sEMG dataset (128-channel, Unicamp Neural Engineering Lab) as the reference dataset, and Xilinx Zybo and ZCU boards as the target hardware.*

---

### S.0 The role of the motion-capture camera: fixed ground truth, not a downstream check

Before discussing algorithm selection, it's worth being precise about what the Vicon motion-capture system actually does in this dataset, since it shapes everything that follows.

The camera is not a validation step that happens *after* the model makes a prediction — it supplies the **ground-truth label used throughout the entire pipeline**, both training and testing:

- **Training:** the model learns from paired data — (HD-sEMG signal, joint angle measured by the camera). Without the camera's measurement, there is nothing to teach the model to predict.
- **Testing/evaluation:** afterward, the model's prediction is compared against the same camera-measured angle to score performance (e.g. the R² figures reported in the dataset paper).

Critically, the camera's measurement is **recorded independently of, and prior to, any model prediction** — it is not influenced in any way by what the model later guesses. This independence is what makes the evaluation valid: if the label were somehow derived from or adjusted based on the model's output, that would constitute data leakage, and any reported accuracy would be meaningless.

**Practical implication:** the camera exists only during dataset collection, to generate reliable labels for research use. It does not travel into a deployed system. In an FPGA-based implementation (or any real-world use), there is no camera — only the HD-sEMG signal is available as input. The camera's role is strictly to make offline training and evaluation possible; the FPGA implementation must operate using EMG alone, exactly as the "surviving candidate pool" in S.3 assumes.

---

### S.0.1 Dataset characterization — fundamental properties and their analytical impact

This subsection isolates the dataset's core properties and states explicitly how each one shapes the analysis and algorithm decisions made throughout this document.

| Property | What it is | How it shapes the analysis |
|---|---|---|
| **128-channel HD-sEMG** | Two 8×8 electrode grids (64 channels each) over the EDC and FDS forearm muscles | Drives criterion 2 (acquisition density) in S.1 — favors spatially-aware methods (Tier 2/3 in S.3) and disfavors sparse-channel pipelines (excluded in S.2) |
| **Continuous sinusoidal movement (not discrete gestures)** | Finger joints move back and forth at a controlled frequency (0.5 Hz used in the paper's analysis), rather than holding fixed poses | Makes this a **regression** problem (predicting a continuously varying joint angle) rather than a **classification** problem (predicting one of a fixed set of labels) — this is why discrete-classification pipelines were excluded in S.2, and why regression-capable models dominate the surviving pool in S.3 |
| **Vicon motion-capture ground truth** | 3D optical tracking of hand markers, converted to joint angles, recorded independently of and prior to any model prediction | Establishes the label used for both training and evaluation (detailed in S.0); confirms the task is regression, not classification, since the label itself is a continuous measurement, not a category |
| **21 healthy subjects, single session, fixed arm posture** | No amputees, no repositioning of electrodes across days, no varying posture within the recorded data | Drives criteria 5 and 6 (data availability, robustness) in S.1 — this is why data-hungry architectures (Transformers, deep hybrids) and shift-invariance-motivated architectures were excluded in S.2: the data doesn't have the variation needed to justify or validate either |
| **Task set skewed toward thumb movements** (5 of 8 patterns involve the thumb) | Reflects the higher anatomical complexity of thumb control relative to other fingers | Not an algorithm-selection factor directly, but relevant to expected accuracy: thumb-angle prediction is consistently the hardest target in the underlying task, regardless of model choice — worth anticipating when interpreting per-finger results later in the project |

---

### S.1 There is no universal "best" algorithm — selection is multi-criteria

The literature is explicit that model choice cannot be reduced to a single dataset or a single accuracy number:

- Lima et al. (Ref. 7) state there is **"no universal consensus or standard method for optimized feature extraction or classification."**
- Fathi et al. (Ref. 10) formalize this as **"The Application Dependency Rule": "There is no universally superior classifier; selecting the ideal model depends entirely on the design constraints of the specific target EMG application."**

Instead, the choice is driven by seven criteria that recur across the references:

| # | Criterion | What it constrains | Key references |
|:---:|---|---|---|
| 1 | **Nature of the task** (steady-state vs. transient/dynamic) | Whether classical feature-based ML or DL feature-learning is the natural fit | Li et al. (Ref. 6, §6.1) |
| 2 | **Acquisition density** (sparse vs. HD-sEMG) | Whether manual feature engineering is required, or spatial DL becomes viable | Li et al. (Ref. 6, §6.2); Tam et al. (Ref. 14) |
| 3 | **Hardware/energy budget of the target** | Which model families are even candidates (MCU-class vs. edge-accelerator-class) | Piyathilaka et al. (Ref. 1, §1.5); Fathi et al. (Ref. 10, "Cost-Selection Principle") |
| 4 | **Latency tolerance of the application** | Whether windowed features (with inherent delay) are acceptable, or instantaneous/image-based inference is required | Tam et al. (Ref. 14, §14.1) |
| 5 | **Availability/diversity of training data** | Whether the source dataset alone is sufficient, or transfer learning/pretraining is needed | Lu et al. (Ref. 15, §15.1, §15.4) |
| 6 | **Required robustness** (electrode shift, inter-session/inter-subject variance) | Whether shift-invariance or normalization strategies must be built into the architecture | Li et al. (Ref. 6, §6.6); Tam et al. (Ref. 14, §14.2) |
| 7 | **Quality of the feature set feeding the model** | Whether the *classifier* choice or the *feature* choice is the actual bottleneck | Souza (Ref. 11, §11.3): **"Selecting the correct feature set is mathematically more critical to control system performance than the choice of classification technique itself."** |

For a project without a fixed deployment target or end user, criteria 1, 2, 5, and 6 are effectively **fixed by the choice of dataset** (they describe properties of the data itself), while criteria 3 and 4 (hardware budget, latency) become the actual **independent variables** the FPGA implementation work is designed to explore.

---

### S.2 Algorithms reasonably excluded given the dataset's fixed criteria

Applying criteria 1, 2, 5, and 6 to the Molinari & Elias HD-sEMG dataset (21 healthy subjects, single session, fixed posture, continuous sinusoidal finger-kinematics regression, 128-channel spatial density) narrows the field before any FPGA-cost analysis is applied:

| Excluded / de-prioritized | Reasoning | Reference basis |
|---|---|---|
| **Unsupervised learning** | The dataset provides fully labeled, hardware-synchronized ground truth (Vicon kinematics) — there is no label-scarcity problem to solve, and unsupervised methods are flagged as unstable under distribution shift | Li et al. (Ref. 6, §6.4); Lima et al. (Ref. 7, §7.1) |
| **Transformers / very deep hybrid CNN-RNN stacks** | Small, fairly homogeneous dataset (single session, single posture); these architectures are data-hungry and were already flagged as computationally excessive for edge use independent of data volume | Piyathilaka et al. (Ref. 1, §1.4); precedent for the data-hungry side of Fathi et al. (Ref. 10, §10.2 "Parametric Deep Learning Tax") |
| **Shift-invariance-oriented architectures as a primary justification** (e.g. Tam et al.'s spatial shift-invariant CNN design) | The dataset has one electrode placement per session with no repositioning/multi-day data to validate shift-robustness against — the architectural motivation isn't exercised by this data | Tam et al. (Ref. 14, §14.2) |
| **Sparse-channel-oriented pipelines** (e.g. Mendes et al.'s armband-based SFS approach) | Technically usable, but wastes the dataset's defining feature — 128-channel spatial density — which is exactly what spatially-aware methods (e.g. MLD-BFM, or HD-sEMG-CNN approaches) are built to exploit | Mendes et al. (Ref. 2); Li et al. (Ref. 6, §6.2) |
| **Pure discrete-classification pipelines** (e.g. Pérez-Reynoso's 4-class if-else MNN) | The dataset's ground truth is continuous finger-joint angles (regression), not discrete gesture labels — a format mismatch, not a hardware or accuracy issue | Pérez-Reynoso et al. (Ref. 9, §9.4) |

---

### S.3 Surviving candidate pool, organized by computational tier

What remains is a three-tier spread, chosen specifically to give a genuine accuracy-vs-cost range for hardware comparison rather than to chase a single best accuracy number.

**Tier 1 — Classical ML on hand-crafted / spatial features (cheapest)**

| Model | Fit rationale | Reference basis |
|---|---|---|
| Ridge / Lasso regression | Closed-form solution, minimal hardware footprint, matches the continuous-regression task directly | Piyathilaka et al. (Ref. 1, §1.5); dataset paper (Molinari & Elias) |
| Decision Tree | Best accuracy-per-cost classical option identified in the literature | Fathi et al. (Ref. 10, §10.3 — 95.46% accuracy, lowest training/testing cost) |
| Random Forest | Higher accuracy ceiling than Decision Tree, at higher structural cost | Fathi et al. (Ref. 10, §10.3 — 98.16% accuracy, heaviest node count); Souza (Ref. 11, whole thesis) |
| Shallow MLP, k-NN | Already validated directly on this dataset and this exact regression task | Dataset paper (Molinari & Elias) |

**Tier 2 — Classical ML + spatially-aware features (mid-range)**

Pairing Tier 1 models with spatial descriptors (e.g. MLD-BFM, or simpler RMS/MAV-WL computed across the HD-sEMG grid) exploits criterion 2 (128-channel density) without requiring DL to do so — a middle ground the "sparse-channel" exclusion in S.2 explicitly does not offer.

**Tier 3 — Lightweight CNNs exploiting HD-sEMG spatial structure (most expensive, most FPGA-proven)**

| Model | Fit rationale | Reference basis |
|---|---|---|
| Shallow 2D-CNN over the electrode grid | Treats the 8×8 arrays as spatial images — the natural use of HD-sEMG density | Tam et al. (Ref. 14); Lu et al. (Ref. 15) |
| UL-DSC + CA-GAP (ultra-lightweight depthwise separable CNN) | Purpose-built for FPGA deployment; already benchmarked in hardware on a **ZCU102** — directly reproducible on the available ZCU board | Guo et al. (Ref. 13, §13.3–13.4: 5.0k parameters, 0.026 MB, 41.9 µs inference, 0.317 W) |

**Tier 2.5 — Cautiously included temporal models**

TCNs and shallow GRUs fit the *task* (continuous, dynamic sinusoidal movement is genuinely transient, not steady-state), but must be kept small to remain plausible for either FPGA target:

| Model | Caveat | Reference basis |
|---|---|---|
| TCN | Cited as outperforming classical ML while managing strict power budgets — better edge fit than deep LSTM stacks | Li et al. (Ref. 6, §6.4) |
| Shallow GRU | Aligns with sEMG's time-series nature, but only if kept shallow | Li et al. (Ref. 6, §6.4) |

---

### S.4 Platform layer: Zybo vs. ZCU as a second comparison axis

Both available boards are Xilinx Zynq SoCs (PS + PL), which removes the DSP-less workaround problem Souza (Ref. 11, §11.4) had to engineer around in VHDL — both boards have native DSP slices for multiply-accumulate operations. What differs is resource *scale*, which maps directly onto the three-tier pool above:

| | Zybo (Zynq-7000) | ZCU (Zynq UltraScale+ MPSoC) |
|---|---|---|
| Resource class | Modest — tens of thousands of LUTs, ~100–200 DSP slices, limited BRAM | Large — hundreds of thousands of logic cells, thousands of DSP slices, several MB BRAM |
| Comfortable fit | Tier 1 (Ridge/Decision Tree + simple or spatial features); possibly a very small MLP | All three tiers, including Tier 3 CNNs |
| Literature precedent | Closer to the "MCU-adjacent" row of Piyathilaka's table (Ref. 1, §1.5) | Matches Guo et al.'s exact evaluation board (Ref. 13, §13.4 — ZCU102, 600k logic cells, 2520 DSP slices) — a directly reproducible benchmark |

This turns board choice into a second, independent axis of comparison alongside model architecture:

1. **Zybo as the hard-constraint case** — any Tier 2/3 candidate that doesn't fit comfortably quantifies a real cost, giving a board-enforced version of the accuracy-vs-resource trade-off.
2. **ZCU as the headroom case and reproduction target** — where Guo et al.'s UL-DSC/QDS-CNN architecture (Ref. 13) can be benchmarked against its own published numbers on matching hardware.
3. **Quantization becomes non-optional on Zybo** — 8-bit fixed-point (per Guo et al., Ref. 13, §13.3) or a more aggressive scheme (per Choi's `ap_fixed<22,6>` sweet spot, Ref. 4, §4.3) is likely required just to fit Tier 2/3 candidates with margin; on ZCU it remains a design choice rather than a necessity.