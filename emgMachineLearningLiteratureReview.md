# EMG Signal Processing & Pattern Recognition
### Literature Reference Guide

---

## Contents

- [EMG Signal Processing \& Pattern Recognition](#emg-signal-processing--pattern-recognition)
    - [Literature Reference Guide](#literature-reference-guide)
  - [Contents](#contents)
  - [Reference 1 — Piyathilaka et al., 2026](#reference-1--piyathilaka-et-al-2026)
    - [1.1 System \& hardware trade-offs](#11-system--hardware-trade-offs)
    - [1.2 Signal processing pipeline](#12-signal-processing-pipeline)
    - [1.3 Feature extraction trade-off sheet](#13-feature-extraction-trade-off-sheet)
    - [1.4 ML architecture trade-offs](#14-ml-architecture-trade-offs)
    - [1.5 Embedded constraints \& resource footprint](#15-embedded-constraints--resource-footprint)
    - [1.6 Challenges \& research gaps](#16-challenges--research-gaps)
    - [1.7 Future trends \& emerging solutions](#17-future-trends--emerging-solutions)
  - [Reference 2 — Mendes et al.](#reference-2--mendes-et-al)
    - [2.1 Usability vs. hardware constraints](#21-usability-vs-hardware-constraints)
    - [2.2 Targeted preprocessing](#22-targeted-preprocessing)
    - [2.3 Sequential forward selection (SFS) wrapper](#23-sequential-forward-selection-sfs-wrapper)
    - [2.4 Embedded optimization — LDA synergy](#24-embedded-optimization--lda-synergy)
  - [Reference 3 — Reaz et al., 2006](#reference-3--reaz-et-al-2006)
    - [3.1 Physiological \& mathematical nature of sEMG](#31-physiological--mathematical-nature-of-semg)
    - [3.2 Signal integrity \& noise categories](#32-signal-integrity--noise-categories)
    - [3.3 Conditioning \& segmentation foundations](#33-conditioning--segmentation-foundations)
    - [3.4 Advanced feature extraction techniques](#34-advanced-feature-extraction-techniques)
    - [3.5 Hardware real-time deployment realities](#35-hardware-real-time-deployment-realities)
    - [3.6 Integrated circuit (IC) design constraints](#36-integrated-circuit-ic-design-constraints)
    - [3.7 Methodological comparison matrix](#37-methodological-comparison-matrix)
  - [Reference 4 — Choi, 2023](#reference-4--choi-2023)
    - [4.1 The edge vs. cloud paradigm](#41-the-edge-vs-cloud-paradigm)
    - [4.2 FPGA optimization \& HLS pipelines](#42-fpga-optimization--hls-pipelines)
    - [4.3 Model compression \& quantization trade-offs](#43-model-compression--quantization-trade-offs)
    - [4.4 Current HLS limitations \& future outlook](#44-current-hls-limitations--future-outlook)

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

- **6th-order Butterworth notch filter** — Explicitly used to eliminate 60 Hz powerline interference. The high-order approximation provides a razor-sharp cutoff slope, killing grid hum while preserving vital muscle signal data directly adjacent to it.

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

 ⚠️ **Critical engineering conflict — the notch filter debate:**
While Mendes et al. (Ref. 2) explicitly advocate for a 6th-order notch filter to kill 60 Hz hum, Reaz et al. explicitly warn that **notch filters are not recommended** because they strip away crucial physiological frequency components and distort raw signal peaks.
 *Design choice:* Prefer high-pass filtering over notch filtering unless environmental line noise completely saturates the ADC.

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
1. **Algorithmic network compression** 2. **Computation compression** (e.g., MobileNet, TPU systolic arrays)
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