"""Camada `connectors/`: Adaptadores para frameworks de Deep Learning (Keras) e síntese HLS (`hls4ml`)."""

from emg_loso.connectors.cnn2d_connector import Cnn2dConfig, TopographicCNN2D
from emg_loso.connectors.hls_export_connector import (
    FixedPointSpec,
    FoldedLayerWeights,
    QuantizationReport,
    export_weights_to_hls_header,
    extract_folded_cnn2d_layers,
    extract_folded_mlp_layers,
    inspect_quantization_saturation,
    quantize_ap_fixed,
    simulate_cnn2d_forward,
    simulate_mlp_forward,
)
from emg_loso.connectors.mlp_connector import DeepMLPRegressor, MlpConfig

__all__ = [
    "Cnn2dConfig",
    "DeepMLPRegressor",
    "FixedPointSpec",
    "FoldedLayerWeights",
    "MlpConfig",
    "QuantizationReport",
    "TopographicCNN2D",
    "export_weights_to_hls_header",
    "extract_folded_cnn2d_layers",
    "extract_folded_mlp_layers",
    "inspect_quantization_saturation",
    "quantize_ap_fixed",
    "simulate_cnn2d_forward",
    "simulate_mlp_forward",
]
