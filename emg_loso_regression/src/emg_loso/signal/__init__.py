"""
Subpacote `emg_loso.signal` (Padrão Pipes & Filters):
Estágios matemáticos puros e sem estado (stateless) para condicionamento de sinal,
janelamento causal, extração de descritores temporais e mapeamento topográfico 2D.
"""

from emg_loso.signal.features import (
    FEATURE_NAMES,
    N_FEATURES_PER_CHANNEL,
    compute_mav,
    compute_rms,
    compute_wl,
    extract_time_domain_features,
)
from emg_loso.signal.filtering import apply_highpass_filter, design_highpass_sos
from emg_loso.signal.windowing import (
    compute_causal_window_bounds,
    make_causal_windows_multidof,
    slice_windows_from_stream,
)

__all__ = [
    "FEATURE_NAMES",
    "N_FEATURES_PER_CHANNEL",
    "apply_highpass_filter",
    "compute_causal_window_bounds",
    "compute_mav",
    "compute_rms",
    "compute_wl",
    "design_highpass_sos",
    "extract_time_domain_features",
    "make_causal_windows_multidof",
    "slice_windows_from_stream",
]
