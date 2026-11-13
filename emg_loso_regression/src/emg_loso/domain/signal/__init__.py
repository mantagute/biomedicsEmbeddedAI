"""Subpacote `emg_loso.domain.signal` (Estágios Pipes & Filters puros do domínio)."""

from emg_loso.domain.signal.features import (
    FEATURE_NAMES,
    N_FEATURES_PER_CHANNEL,
    compute_mav,
    compute_rms,
    compute_wl,
    extract_time_domain_features,
)
from emg_loso.domain.signal.filtering import apply_highpass_filter, design_highpass_sos
from emg_loso.domain.signal.topography import (
    EXPECTED_FLAT_DIM,
    N_TOPOGRAPHIC_CHANNELS,
    extract_recording_representations,
    flat_to_topographic_grid,
    topographic_grid_to_flat,
)
from emg_loso.domain.signal.windowing import (
    compute_causal_window_bounds,
    make_causal_windows_multidof,
    slice_windows_from_stream,
)

__all__ = [
    "EXPECTED_FLAT_DIM",
    "FEATURE_NAMES",
    "N_FEATURES_PER_CHANNEL",
    "N_TOPOGRAPHIC_CHANNELS",
    "apply_highpass_filter",
    "compute_causal_window_bounds",
    "compute_mav",
    "compute_rms",
    "compute_wl",
    "design_highpass_sos",
    "extract_recording_representations",
    "extract_time_domain_features",
    "flat_to_topographic_grid",
    "make_causal_windows_multidof",
    "slice_windows_from_stream",
    "topographic_grid_to_flat",
]
