```markdown
# EMG Signal Processing & Pattern Recognition: Literature Reference Guide

---

## In this document

- [Reference 1: Advances in EMG Signal Processing and Pattern Recognition (Piyathilaka et al., 2026)](#reference-1-advances-in-emg-signal-processing-and-pattern-recognition-piyathilaka-et-al-2026)
  - [1.1. System \& Hardware Trade-offs](#11-system--hardware-trade-offs)
  - [1.2. EMG Signal Processing Pipeline](#12-emg-signal-processing-pipeline)
  - [1.3. Feature Extraction Trade-off Sheet](#13-feature-extraction-trade-off-sheet)
  - [1.4. Machine Learning Architecture Trade-offs](#14-machine-learning-architecture-trade-offs)
  - [1.5. Embedded Constraints \& Resource Footprint](#15-embedded-constraints--resource-footprint)
  - [1.6. Current Challenges \& Research Gaps](#16-current-challenges--research-gaps)
  - [1.7. Future Trends \& Emerging Solutions](#17-future-trends--emerging-solutions)
- [Reference 2: Feature Selection of sEMG for Hand Gesture Classification by Armband (Mendes et al.)](#reference-2-feature-selection-of-semg-for-hand-gesture-classification-by-armband-mendes-et-al)
  - [2.1. System Constraints \& Usability Realities](#21-system-constraints--usability-realities)
  - [2.2. Targeted Hardware Preprocessing](#22-targeted-hardware-preprocessing)
  - [2.3. Feature Selection Methodology (Wrapper / SFS Approach)](#23-feature-selection-methodology-wrapper--sfs-approach)
  - [2.4. Embedded Optimization \& Classifier Synergy](#24-embedded-optimization--classifier-synergy)

---

# Reference 1: Advances in EMG Signal Processing and Pattern Recognition (Piyathilaka et al., 2026)

**Citation —** L. Piyathilaka, J.-H. Sul, S. D. Arachchige, A. Jayawardena, and D. Moratuwage, "Advances in EMG Signal Processing and Pattern Recognition: Techniques, Challenges, and Emerging Applications," *Electronics*, vol. 15, no. 3, p. 590, 2026. Available: https://www.mdpi.com/2079-9292/15/3/590

### 1.1. System & Hardware Trade-offs

| Topic | Description |
|:---|:---|
| **Non-stationarity** | sEMG pipelines degrade over time due to electrode shifts, sweat, and fatigue. |
| **Hardware reality** | HD-sEMG fixes spatial shifts but introduces heavy computational loads and energy drains, making it prohibitive for edge/embedded devices. |
| **Classifiers** | Traditional ML (SVM, LDA) is computationally cheap but vulnerable to signal variations. LSTM/GRU handle temporal dynamics much better. |

### 1.2. EMG Signal Processing Pipeline

To make raw muscle data readable by a neural network without draining embedded resources, the signal must follow this strict conditioning flow.

| Step | Stage | Description |
|:---:|:---|:---|
| ⚡ | **Raw sEMG signal** | Highly noisy, affected by skin impedance and sweat. |
| 1 | **Band-pass filtering** | High-pass (5–10 Hz) removes motion artifacts and baseline drift. Low-pass (400–450 Hz) suppresses electrical interference. |
| 2 | **Rectification & conditioning** | Translates negative wave amplitudes into positive values for easier mathematical analysis. |
| 3 | **Feature extraction** | Compresses the data stream into dense mathematical metrics across time, frequency, or non-linear domains. |
| 4 | **Classification / pattern recognition** | Features are fed into the ML or neural network model (e.g., LSTM). |

### 1.3. Feature Extraction Trade-off Sheet

Use this table to rapidly justify feature selection based on embedded constraints and signal robustness.

| Domain | Feature | Complexity | Noise Robustness | Real-Time | Key Justification |
|:---|:---|:---|:---|:---|:---|
| Time (TD) | `MAV` | Low | Low | Excellent | Cheapest to compute, but highly sensitive to electrode shifts and high-amplitude noise. |
| Time (TD) | `RMS` | Low | Medium | Excellent | Best balance of noise robustness and low cost; strong physiological relevance. |
| Time (TD) | `WL` | Low | Medium | Excellent | Captures amplitude variation; robust to scaling but requires solid preprocessing. |
| Frequency (FD) | `MNF` | Medium | High | Good | Tracks fatigue well, but highly sensitive to spectral outliers/noise. |
| Frequency (FD) | `MDF` | Medium | High | Good | Safer choice than MNF; less sensitive to spectral spikes. |
| Time-Freq. (TFD) | `STFT` | High | High | Fair | Simple conceptually, but performs poorly in highly dynamic movements. |
| Time-Freq. (TFD) | `CWT` | Very High | Very High | Poor | Excellent dynamic detail, but demands excessive parameter tuning and compute cost. |
| Time-Freq. (TFD) | `DWT` | Med-High | High | Fair | Most efficient TFD method, but suffers from shift sensitivity and limited scale flexibility. |
| Non-linear | `SampEn` | Medium | Very High | Good | Preferred over ApEn for non-Gaussian EMG complexity; less sensitive to data length. |
| Non-linear | `ApEn` | Medium | High | Good | Handles complexity but is biased for short data segments. |

### 1.4. Machine Learning Architecture Trade-offs

There is a clear methodological transition from conventional handcrafted-feature pipelines to deep and hybrid architectures. However, each introduces distinct engineering trade-offs.

| Architecture | Strengths | Limitations |
|:---|:---|:---|
| **Classical ML** (SVM, LDA) | Fast, highly interpretable. | Weak user generalization; completely dependent on handcrafted features. |
| **CNN** | Strong spatial learning. | Computationally complex; sensitive to electrode shifts unless highly augmented. |
| **RNN** (LSTM, GRU) | Strong temporal modeling. | Limited long-range context; higher inference latency and computationally costly. |
| **Hybrid CNN–RNN** | Best spatio-temporal fusion. | Heavy computation; large parameter counts make embedded deployment difficult without pruning. |
| **Transformers** | Long-range attention. | Very high computational cost; unsuitable for real-time/battery-powered edge systems. |
| **GNNs** | Strong spatial topology. | Complex graph setup required. |
| **Multimodal Fusion** | High robustness (integrates IMU/FMG). | Extra sensors required; increased sync complexity, cost, and power draw. |

### 1.5. Embedded Constraints & Resource Footprint

*This table establishes the hardware limits for embedded neural networks, proving why certain architectures are favored over others in resource-constrained environments.*

| Model Family | Inference Latency | Memory Footprint | Power & Hardware Suitability |
|:---|:---|:---|:---|
| **Classical ML** (TD features + LDA/SVM) | Microseconds (µs) to low ms | **Very Low** (~1 kB) | Excellent for MCU/DSP; guarantees long battery life. |
| **Shallow NN / CNN** | Low milliseconds (ms) | **Low–Moderate** (10–100 kB) | Suitable for standard embedded edge devices. |
| **CNN–RNN Hybrids** | Several ms to 10 ms | **Moderate** (100 kB–1 MB) | Feasible on edge hardware only with careful optimization and pruning. |
| **Transformers / Attention Models** | > 10 ms (sequence-dependent) | **High** (MB range) | Typically requires a dedicated Edge AI accelerator; drains battery quickly. |

### 1.6. Current Challenges & Research Gaps

Despite high performance in controlled laboratory-centric evaluation paradigms, significant roadblocks prevent seamless, out-of-the-lab deployment.

* **Electrode Shift & Non-Stationarity:** Signal structures alter drastically as sensors shift during daily activities. This creates an immense user-usability burden by forcing frequent manual system recalibration.
* **Physiological & Session Variability:** High inter-subject (cross-user) and intra-session variance is driven dynamically by:
  * Fluctuations in skin impedance and subcutaneous fat layers.
  * Variations in muscle fiber composition and changing force levels exerted during movements.
  * Quality of the electrode–skin contact interface over time.
* **Data Constraints:** A persistent lack of standardized, large-scale, public EMG datasets restricts model training scaling, raising ethical, privacy, and user-centric data collection considerations.

### 1.7. Future Trends & Emerging Solutions

To achieve robust out-of-lab performance, modern research is shifting toward decentralized, multi-sensor systems featuring real-time adaptation.

| Paradigm | Emerging Solution | Engineering Value for Projects |
|:---|:---|:---|
| **On-Device Adaptation** | Lightweight Edge AI with stable, incremental updating strategies. | Solves calibration-free operation by continuously fighting sensor drift without cloud dependencies. Real-time, safe, and secure inference. |
| **Model Optimization** | Quantization, model compression, and structural pruning. | Directly addresses hardware barriers, adapting heavy spatial/temporal networks down to wearable battery envelopes. |
| **Generalization Frameworks** | Transfer Learning and Domain Adaptation techniques. | Mitigates cross-session performance degradation, adapting pre-trained models to new subjects with minimal user calibration data. |
| **Multimodal Fusion** | Co-registering surface EMG with IMU (Inertial), FMG (Force), or optical sensors. | Stabilizes gesture predictions under highly dynamic arm positions and muscle fatigue, bypassing traditional single-modality limits. |
| **Data Expansion** | Synthetic EMG data generation frameworks. | Bypasses traditional data scarcity and privacy bottlenecks by training models on robust, simulated physiological waves. |

---

# Reference 2: Feature Selection of sEMG for Hand Gesture Classification by Armband (Mendes et al.)

**Citation —** J. Mendes et al., "Comparative Analysis Among Feature Selection of sEMG Signal for Hand Gesture Classification by Armband," *UTFPR*.

---

### 2.1. Usability vs. Hardware Constraints
* **The Placement Problem:** Precise muscle targeting yields high accuracy, but everyday users cannot accurately locate specific muscle groups.
* **The Armband Solution:** Uses fixed, equidistant sensor arrays. This shifts the engineering burden away from perfect physical placement and onto the processing algorithms.

---

### 2.2. Targeted Preprocessing
* **6th-Order Butterworth Notch Filter:** Explicitly used to eliminate 60 Hz powerline interference. 
* **Justification:** The high-order (6th) approximation provides a razor-sharp cutoff slope, killing grid hum while preserving vital muscle signal data directly next to it.

---

### 2.3. Sequential Forward Selection (SFS) Feature Wrapper
* **How it works:** Tests features individually to find a "winner," then iteratively tests combinations with remaining features to build an optimal subset.

| Approach | Engineering Pros | Engineering Cons |
|:---|:---|:---|
| **SFS Wrapper** | Tailors a highly accurate, custom feature subset optimized *directly* for your specific classifier. | High computational overhead during the initial offline training/search phase. |

---

### 2.4. Embedded Optimization (LDA Synergy)
* **Dimensionality Reduction:** Trimming the feature pool is mandatory before embedding code onto a microcontroller.
* **LDA Boundary Synergy:** Pairing highly correlated, mutually supportive features explicitly strengthens the mathematical decision boundaries of Linear Discriminant Analysis (LDA), leading to cleaner cluster separation.
* **The Bottom Line:** Removing redundant attributes lets you achieve **identical classification accuracy for core gestures while radically lowering the computational load** on your embedded hardware.