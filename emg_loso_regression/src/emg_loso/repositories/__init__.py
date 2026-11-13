"""Camada `repositories/`: Acesso a arquivos CSV brutos e persistência de cache (.npz)."""

from emg_loso.repositories.cache_repository import (
    NpzSubjectCacheRepository,
    build_subject_dataset_from_csv,
    ensure_subjects_cached,
)
from emg_loso.repositories.csv_repository import (
    discover_subject_recordings,
    load_angles_dataframe,
    load_emg_array,
    parse_recording_id,
)

__all__ = [
    "NpzSubjectCacheRepository",
    "build_subject_dataset_from_csv",
    "discover_subject_recordings",
    "ensure_subjects_cached",
    "load_angles_dataframe",
    "load_emg_array",
    "parse_recording_id",
]
