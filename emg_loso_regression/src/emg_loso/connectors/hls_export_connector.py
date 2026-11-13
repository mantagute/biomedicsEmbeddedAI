"""
connectors/hls_export_connector.py
----------------------------------
Plugin/Adaptador de Exportação para Hardware (FPGA / Vitis HLS).

Responsabilidades:
1. Folding matemático de `BatchNormalization` nos pesos de `Dense` e `Conv2D`
   (`W_folded`, `b_folded`), eliminando divisões e parâmetros extras na FPGA.
2. Simulação bit-accurate de quantização `ap_fixed<total_bits, int_bits>` em
   NumPy puro (com verificação de overflow/saturação) para validar a degradação
   de RMSE / R² antes da síntese RTL.
3. Geração de header C++ (`.h`) com arrays `static const` em `ap_fixed` prontos
   para implementação direta em Vitis/Vivado HLS (sem depender do `hls4ml`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from tensorflow.keras import Model, layers

from emg_loso.connectors.cnn2d_connector import TopographicCNN2D
from emg_loso.connectors.mlp_connector import DeepMLPRegressor
from emg_loso.controllers.loso_controller import FeatureStandardizer
from emg_loso.domain.signal.topography import flat_to_topographic_grid


@dataclass(frozen=True, slots=True)
class FixedPointSpec:
    """Especificação de ponto fixo estilo Xilinx/AMD `ap_fixed<W, I>`."""

    total_bits: int = 16
    int_bits: int = 6
    round_to_nearest: bool = True

    def __post_init__(self) -> None:
        if self.total_bits <= 1 or self.int_bits < 1 or self.int_bits >= self.total_bits:
            raise ValueError(
                f"Spec ap_fixed<{self.total_bits}, {self.int_bits}> inválida."
            )

    @property
    def frac_bits(self) -> int:
        return self.total_bits - self.int_bits

    @property
    def step(self) -> float:
        return float(2.0 ** (-self.frac_bits))

    @property
    def min_val(self) -> float:
        return float(-(2.0 ** (self.int_bits - 1)))

    @property
    def max_val(self) -> float:
        return float((2.0 ** (self.int_bits - 1)) - self.step)


@dataclass(frozen=True, slots=True)
class FoldedLayerWeights:
    """Pesos e bias de uma camada linear/convolucional após folding de BN."""

    name: str
    kind: str  # "dense" | "conv2d"
    weight: np.ndarray
    bias: np.ndarray
    activation: str  # "relu" | "linear"


@dataclass(frozen=True, slots=True)
class QuantizationReport:
    """Relatório de faixa dinâmica e saturação para um conjunto de tensores."""

    spec: FixedPointSpec
    max_abs_observed: float
    n_saturated_weights: int
    total_weights: int


def quantize_ap_fixed(x: np.ndarray, spec: FixedPointSpec) -> np.ndarray:
    """Simula quantização `ap_fixed<W, I, AP_RND/AP_TRN, AP_SAT>` em float32."""
    step = spec.step
    scaled = x.astype(np.float64) / step
    q_int = np.round(scaled) if spec.round_to_nearest else np.floor(scaled)
    q_val = q_int * step
    return np.clip(q_val, spec.min_val, spec.max_val).astype(np.float32)


def _fold_bn_params(
    gamma: np.ndarray,
    beta: np.ndarray,
    moving_mean: np.ndarray,
    moving_var: np.ndarray,
    epsilon: float,
    original_bias: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula fator de escala `s` e bias fundido `b_fold` de uma BatchNormalization."""
    inv_std = 1.0 / np.sqrt(moving_var.astype(np.float64) + float(epsilon))
    scale = gamma.astype(np.float64) * inv_std
    b0 = (
        np.zeros_like(moving_mean, dtype=np.float64)
        if original_bias is None
        else original_bias.astype(np.float64)
    )
    b_fold = beta.astype(np.float64) + scale * (b0 - moving_mean.astype(np.float64))
    return scale.astype(np.float32), b_fold.astype(np.float32)


def fold_dense_bn(
    dense_layer: layers.Dense,
    bn_layer: layers.BatchNormalization,
    activation: str = "relu",
) -> FoldedLayerWeights:
    """Funde `Dense -> BatchNormalization` em `(W_fold, b_fold)` equivalentes."""
    weights = dense_layer.get_weights()
    w = weights[0]
    b0 = weights[1] if len(weights) > 1 else None
    gamma, beta, mean, var = bn_layer.get_weights()
    scale, b_fold = _fold_bn_params(gamma, beta, mean, var, bn_layer.epsilon, b0)
    w_fold = (w * scale[None, :]).astype(np.float32)
    return FoldedLayerWeights(
        name=dense_layer.name,
        kind="dense",
        weight=w_fold,
        bias=b_fold,
        activation=activation,
    )


def fold_conv2d_bn(
    conv_layer: layers.Conv2D,
    bn_layer: layers.BatchNormalization,
    activation: str = "relu",
) -> FoldedLayerWeights:
    """Funde `Conv2D -> BatchNormalization` em `(W_fold, b_fold)` equivalentes."""
    weights = conv_layer.get_weights()
    w = weights[0]
    b0 = weights[1] if len(weights) > 1 else None
    gamma, beta, mean, var = bn_layer.get_weights()
    scale, b_fold = _fold_bn_params(gamma, beta, mean, var, bn_layer.epsilon, b0)
    w_fold = (w * scale[None, None, None, :]).astype(np.float32)
    return FoldedLayerWeights(
        name=conv_layer.name,
        kind="conv2d",
        weight=w_fold,
        bias=b_fold,
        activation=activation,
    )


def extract_folded_mlp_layers(plugin: DeepMLPRegressor) -> list[FoldedLayerWeights]:
    """Extrai todas as camadas do `DeepMLPRegressor` com BN já fundida."""
    model: Model = plugin.model
    folded: list[FoldedLayerWeights] = []
    n_hidden = len(plugin.cfg.hidden_units)

    for idx in range(1, n_hidden + 1):
        dense = model.get_layer(f"fc{idx}")
        bn = model.get_layer(f"bn{idx}")
        folded.append(fold_dense_bn(dense, bn, activation="relu"))

    head = model.get_layer("regression_head")
    w_head, b_head = head.get_weights()
    folded.append(
        FoldedLayerWeights(
            name="regression_head",
            kind="dense",
            weight=w_head.astype(np.float32),
            bias=b_head.astype(np.float32),
            activation="linear",
        )
    )
    return folded


def extract_folded_cnn2d_layers(plugin: TopographicCNN2D) -> list[FoldedLayerWeights]:
    """Extrai todas as camadas do `TopographicCNN2D` com BN já fundida."""
    model: Model = plugin.model
    conv1 = fold_conv2d_bn(model.get_layer("conv1"), model.get_layer("bn1"), activation="relu")
    conv2 = fold_conv2d_bn(model.get_layer("conv2"), model.get_layer("bn2"), activation="relu")
    fc_lat = fold_dense_bn(
        model.get_layer("fc_latent"),
        model.get_layer("bn_latent"),
        activation="relu",
    )
    head = model.get_layer("regression_head")
    w_head, b_head = head.get_weights()
    head_folded = FoldedLayerWeights(
        name="regression_head",
        kind="dense",
        weight=w_head.astype(np.float32),
        bias=b_head.astype(np.float32),
        activation="linear",
    )
    return [conv1, conv2, fc_lat, head_folded]


def inspect_quantization_saturation(
    folded_layers: Iterable[FoldedLayerWeights],
    spec: FixedPointSpec,
) -> QuantizationReport:
    """Verifica se algum peso ou bias excede a faixa representável de `ap_fixed`."""
    max_abs = 0.0
    n_sat = 0
    total = 0
    for layer in folded_layers:
        for arr in (layer.weight, layer.bias):
            abs_arr = np.abs(arr)
            max_abs = max(max_abs, float(np.max(abs_arr)))
            n_sat += int(np.sum((arr < spec.min_val) | (arr > spec.max_val)))
            total += int(arr.size)
    return QuantizationReport(
        spec=spec,
        max_abs_observed=max_abs,
        n_saturated_weights=n_sat,
        total_weights=total,
    )


def simulate_mlp_forward(
    x_eval: np.ndarray,
    folded_layers: list[FoldedLayerWeights],
    spec: FixedPointSpec | None = None,
) -> np.ndarray:
    """Forward pass NumPy (float32 ou quantizado `ap_fixed`) para o `DeepMLPRegressor`."""
    x = x_eval.astype(np.float32, copy=False)
    if spec is not None:
        x = quantize_ap_fixed(x, spec)

    for layer in folded_layers:
        w = quantize_ap_fixed(layer.weight, spec) if spec is not None else layer.weight
        b = quantize_ap_fixed(layer.bias, spec) if spec is not None else layer.bias
        x = x @ w + b
        if layer.activation == "relu":
            x = np.maximum(x, np.float32(0.0))
        if spec is not None:
            x = quantize_ap_fixed(x, spec)
    return x.astype(np.float32, copy=False)


def _conv2d_same_3x3(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Convolução 2D 3x3 padding='same' vetorizada no lote `(N, H, W, Cin)`."""
    n, h, w, c_in = x.shape
    c_out = kernel.shape[3]
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="constant")
    out = np.zeros((n, h, w, c_out), dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            patch = padded[:, dy : dy + h, dx : dx + w, :]
            out += np.tensordot(patch, kernel[dy, dx], axes=([3], [0]))
    out += bias[None, None, None, :]
    return out


def _maxpool_2x2(x: np.ndarray) -> np.ndarray:
    """MaxPooling 2x2 stride 2 para tensores `(N, H, W, C)` com H, W pares."""
    n, h, w, c = x.shape
    return x.reshape(n, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))


def simulate_cnn2d_forward(
    x_flat_eval: np.ndarray,
    folded_layers: list[FoldedLayerWeights],
    spec: FixedPointSpec | None = None,
) -> np.ndarray:
    """Forward pass NumPy (float32 ou `ap_fixed`) para o `TopographicCNN2D`."""
    conv1, conv2, fc_lat, head = folded_layers
    x = flat_to_topographic_grid(x_flat_eval)
    if spec is not None:
        x = quantize_ap_fixed(x, spec)

    def _w(layer: FoldedLayerWeights) -> tuple[np.ndarray, np.ndarray]:
        if spec is None:
            return layer.weight, layer.bias
        return quantize_ap_fixed(layer.weight, spec), quantize_ap_fixed(layer.bias, spec)

    w1, b1 = _w(conv1)
    x = np.maximum(_conv2d_same_3x3(x, w1, b1), np.float32(0.0))
    if spec is not None:
        x = quantize_ap_fixed(x, spec)
    x = _maxpool_2x2(x)

    w2, b2 = _w(conv2)
    x = np.maximum(_conv2d_same_3x3(x, w2, b2), np.float32(0.0))
    if spec is not None:
        x = quantize_ap_fixed(x, spec)

    # GlobalAveragePooling2D sobre 4x4 (divisão exata por 16 = >> 4 em binário)
    x = np.mean(x, axis=(1, 2), dtype=np.float32)
    if spec is not None:
        x = quantize_ap_fixed(x, spec)

    wl, bl = _w(fc_lat)
    x = np.maximum(x @ wl + bl, np.float32(0.0))
    if spec is not None:
        x = quantize_ap_fixed(x, spec)

    wh, bh = _w(head)
    out = x @ wh + bh
    if spec is not None:
        out = quantize_ap_fixed(out, spec)
    return out.astype(np.float32, copy=False)


def _format_c_array(name: str, arr: np.ndarray, c_type: str) -> str:
    """Formata um tensor NumPy achatado em declaração `static const` C++."""
    flat = arr.ravel()
    dims = "".join(f"[{d}]" for d in arr.shape)
    vals = ", ".join(f"{float(v):.8f}f" for v in flat)
    return f"static const {c_type} {name}{dims} = {{{vals}}};\n"


def export_weights_to_hls_header(
    out_path: Path,
    folded_layers: list[FoldedLayerWeights],
    spec: FixedPointSpec,
    standardizer: FeatureStandardizer | None = None,
    guard_name: str = "EMG_MODEL_WEIGHTS_H",
) -> Path:
    """Gera header C++ `.h` com pesos folded + standardizer para Vitis/Vivado HLS."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f"#ifndef {guard_name}",
        f"#define {guard_name}",
        "",
        "#include <ap_fixed.h>",
        "",
        f"typedef ap_fixed<{spec.total_bits}, {spec.int_bits}, AP_RND, AP_SAT> hls_weight_t;",
        "",
    ]

    if standardizer is not None and standardizer.mean_ is not None and standardizer.scale_ is not None:
        inv_scale = (1.0 / standardizer.scale_).astype(np.float32)
        lines.append("// FeatureStandardizer: x_norm = (x - MEAN) * INV_SCALE")
        lines.append(_format_c_array("STD_MEAN", standardizer.mean_, "hls_weight_t"))
        lines.append(_format_c_array("STD_INV_SCALE", inv_scale, "hls_weight_t"))

    for layer in folded_layers:
        upper = layer.name.upper()
        lines.append(f"// Camada {layer.name} ({layer.kind}, act={layer.activation})")
        lines.append(_format_c_array(f"{upper}_W", layer.weight, "hls_weight_t"))
        lines.append(_format_c_array(f"{upper}_B", layer.bias, "hls_weight_t"))

    lines.append(f"#endif // {guard_name}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
