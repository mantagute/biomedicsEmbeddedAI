# EMG hls4ml MVP

Minimal pipeline for exploring HD-sEMG recordings and hand kinematics before
moving to small regression models and hls4ml.

The first week focuses on one reproducible path:

```text
CSV files -> pandas tables -> causal EMG windows -> numpy feature matrix -> baseline model
```

## Project Layout

```text
emg_hls4ml_mvp/
├── data/
│   ├── raw/          # local dataset files
│   └── processed/    # generated arrays and artifacts
├── notebooks/        # exploratory notebooks
├── src/              # reusable Python implementation
└── requirements.txt
```

## Environment Setup

Create a virtual environment from the project folder:

```bash
cd emg_hls4ml_mvp
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency versions are bounded in `requirements.txt`. The lower bound
records the feature/API generation used by this project, while the upper bound
avoids silent major-version upgrades that could change behavior.

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name emg-hls4ml-mvp --display-name "EMG hls4ml MVP"
```

## Running the Notebook

Start Jupyter from the project folder:

```bash
jupyter notebook
```

Open:

```text
notebooks/01_inspect_sub001.ipynb
```

Select the kernel:

```text
EMG hls4ml MVP
```

## Current Dataset Assumption

The first notebook expects this recording to exist:

```text
data/raw/Sub001/HD_sEMG/Sub001_1_05_450_0.csv
data/raw/Sub001/HandKinematics/Angles/Sub001_1_05_450_0.csv
```

The current target column is:

```text
index_z
```

## Notes

The dataset files are local artifacts and should stay out of Git. Generated
outputs under `data/processed/` should also be treated as reproducible artifacts.

TensorFlow, QKeras, and hls4ml are intentionally not included yet. They will be
added when the MVP reaches the neural model and hardware-conversion steps.
