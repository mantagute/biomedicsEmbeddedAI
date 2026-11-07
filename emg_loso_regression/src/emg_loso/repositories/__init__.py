"""Camada `repositories/`: Acesso a arquivos CSV brutos e persistência de cache (.npz)."""

from emg_loso.repositories.csv_repository import (
    discover_subject_recordings,
    load_angles_dataframe,
    load_emg_array,
    parse_recording_id,
)

__all__ = [
    "discover_subject_recordings",
    "load_angles_dataframe",
    "load_emg_array",
    "parse_recording_id",
]
