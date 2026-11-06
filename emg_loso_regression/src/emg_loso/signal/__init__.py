"""
Subpacote `emg_loso.signal` (Padrão Pipes & Filters):
Estágios matemáticos puros e sem estado (stateless) para condicionamento de sinal,
janelamento causal, extração de descritores temporais e mapeamento topográfico 2D.
"""

from emg_loso.signal.filtering import apply_highpass_filter, design_highpass_sos
from emg_loso.signal.windowing import (
    compute_causal_window_bounds,
    make_causal_windows_multidof,
    slice_windows_from_stream,
)

__all__ = [
    "apply_highpass_filter",
    "compute_causal_window_bounds",
    "design_highpass_sos",
    "make_causal_windows_multidof",
    "slice_windows_from_stream",
]
