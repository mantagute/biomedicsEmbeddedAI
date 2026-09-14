from __future__ import annotations

from pathlib import Path

import pandas as pd


# Dataset clocks: EMG is the fast signal; kinematics is the slower label signal.
EMG_SAMPLE_RATE_HZ = 2052.52
KINEMATICS_SAMPLE_RATE_HZ = 100.0

# Explicit column names turn the Vicon CSV into a named table.
ANGLE_COLUMNS = [
    "frame",
    "sub_frame",
    "thumb_x",
    "thumb_y",
    "thumb_z",
    "index_x",
    "index_y",
    "index_z",
    "middle_x",
    "middle_y",
    "middle_z",
    "ring_x",
    "ring_y",
    "ring_z",
    "little_x",
    "little_y",
    "little_z",
]


def load_emg_csv(path: str | Path) -> pd.DataFrame:
    """Load one HD-sEMG CSV as a pandas table: rows=time, columns=channels."""
    path = Path(path)
    columns = [f"ch_{i:03d}" for i in range(1, 129)]

    return pd.read_csv(
        path,
        header=None,
        names=columns,
        dtype="float32",
        skipinitialspace=True,
    )


def load_angles_csv(path: str | Path) -> pd.DataFrame:
    """Load one Vicon Angles CSV as a pandas table with named angle columns."""
    path = Path(path)

    return pd.read_csv(
        path,
        header=None,
        names=ANGLE_COLUMNS,
        skiprows=5,
        dtype="float32",
        skipinitialspace=True,
    )


def recording_paths(
    data_root: str | Path,
    subject: str,
    recording_id: str,
) -> tuple[Path, Path]:
    """Return the matching EMG and Angles file paths for one recording id."""
    subject_root = Path(data_root) / "raw" / subject
    filename = f"{recording_id}.csv"

    emg_path = subject_root / "HD_sEMG" / filename
    angles_path = subject_root / "HandKinematics" / "Angles" / filename

    if not emg_path.exists():
        raise FileNotFoundError(f"Missing EMG file: {emg_path}")
    if not angles_path.exists():
        raise FileNotFoundError(f"Missing Angles file: {angles_path}")

    return emg_path, angles_path
