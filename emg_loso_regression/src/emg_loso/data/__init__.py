"""
Subpacote `emg_loso.data`:
Adaptadores de Entrada/Saída (I/O), controle de qualidade de gravações e
persistência de cache compacto (.npz) para os 21 sujeitos do protocolo LOSO.
"""

from emg_loso.data.io import (
    RecordingInfo,
    discover_subject_recordings,
    load_angles_dataframe,
    load_emg_array,
    parse_recording_id,
)

__all__ = [
    "RecordingInfo",
    "discover_subject_recordings",
    "load_angles_dataframe",
    "load_emg_array",
    "parse_recording_id",
]
