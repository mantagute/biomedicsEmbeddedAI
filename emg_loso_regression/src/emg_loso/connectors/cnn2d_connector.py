"""
connectors/cnn2d_connector.py
-----------------------------
Plugin 2 do Microkernel (Camada `connectors/`): Topographic 2D CNN Regressor em Keras.

Arquitetura Hardware-Aware (compatível tanto com `hls4ml` quanto com HLS C++ manual):
- Entrada: vetor plano `(384,)`, convertido internamente via
  `flat_to_topographic_grid` para o tensor espacial `(8, 8, 6)`
  (`2 arrays 8x8 [EDC, FDS] x 3 features [RMS, MAV, WL]`).
- Backbone `f_θ`:
  `Conv2D(16, 3x3, padding='same', use_bias=False) -> BN -> ReLU -> MaxPool(2x2)`
  `-> Conv2D(32, 3x3, padding='same', use_bias=False) -> BN -> ReLU`
  `-> GlobalAveragePooling2D()` (divisão por 16 = bit-shift `>> 4` em ponto fixo)
  `-> Dense(32, use_bias=False) -> BN -> ReLU`
- Regression Head: `Dense(5, linear)` prevendo os 5 dedos simultaneamente.

Nota de Engenharia FPGA:
Por usar apenas blocos padrão (`Conv2D 3x3`, `MaxPool 2x2`, `GlobalAvgPool`, `Dense`),
os parâmetros de `BatchNormalization` podem ser fundidos (*folded*) nos pesos das
camadas anteriores e exportados diretamente para HLS C++ modular (`ap_fixed`), caso
o `hls4ml` apresente limitações na placa alvo (ex.: Zybo Z7-20).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.linear_model import Ridge
import tensorflow as tf
from tensorflow.keras import Model, callbacks, layers, optimizers

from emg_loso.config import GRID_COLS, GRID_ROWS, N_DOFS
from emg_loso.domain.signal.topography import (
    N_TOPOGRAPHIC_CHANNELS,
    flat_to_topographic_grid,
)


@dataclass(frozen=True, slots=True)
class Cnn2dConfig:
    """Hiperparâmetros do plugin `TopographicCNN2D`."""

    conv_filters: tuple[int, int] = (16, 32)
    kernel_size: int = 3
    latent_dim: int = 32
    dropout_rate: float = 0.2
    epochs: int = 25
    batch_size: int = 256
    learning_rate: float = 1e-3
    validation_split: float = 0.1
    early_stopping_patience: int = 5
    adaptation_mode: Literal["ridge", "adam"] = "ridge"
    ridge_alpha: float = 10.0
    calib_epochs: int = 10
    calib_lr: float = 1e-3
    seed: int = 42


class TopographicCNN2D:
    """Implementação do contrato `AdaptableRegressorPort` usando CNN 2D topográfica."""

    def __init__(self, cfg: Cnn2dConfig = Cnn2dConfig()) -> None:
        self.cfg = cfg
        tf.keras.utils.set_random_seed(cfg.seed)
        self.model, self.backbone, self.head = self._build_model()

    @property
    def name(self) -> str:
        f1, f2 = self.cfg.conv_filters
        return f"TopographicCNN2D({f1}-{f2}-lat{self.cfg.latent_dim}, adapt={self.cfg.adaptation_mode})"

    def _build_model(self) -> tuple[Model, Model, layers.Dense]:
        inputs = layers.Input(
            shape=(GRID_ROWS, GRID_COLS, N_TOPOGRAPHIC_CHANNELS),
            name="emg_grid_in",
        )
        f1, f2 = self.cfg.conv_filters
        k = self.cfg.kernel_size

        x = layers.Conv2D(f1, k, padding="same", use_bias=False, name="conv1")(inputs)
        x = layers.BatchNormalization(name="bn1")(x)
        x = layers.Activation("relu", name="relu1")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

        x = layers.Conv2D(f2, k, padding="same", use_bias=False, name="conv2")(x)
        x = layers.BatchNormalization(name="bn2")(x)
        x = layers.Activation("relu", name="relu2")(x)

        # Em entrada 8x8 -> MaxPool(2x2) -> 4x4, GlobalAveragePooling2D reduz 4x4=16 pontos
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        if self.cfg.dropout_rate > 0.0:
            x = layers.Dropout(self.cfg.dropout_rate, name="drop_gap")(x)

        x = layers.Dense(self.cfg.latent_dim, use_bias=False, name="fc_latent")(x)
        x = layers.BatchNormalization(name="bn_latent")(x)
        x = layers.Activation("relu", name="relu_latent")(x)

        backbone = Model(inputs=inputs, outputs=x, name="cnn2d_backbone")
        head = layers.Dense(N_DOFS, activation="linear", name="regression_head")
        outputs = head(x)
        full_model = Model(inputs=inputs, outputs=outputs, name="topographic_cnn2d")
        return full_model, backbone, head

    def fit_backbone(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Converte `x_train` para `(N, 8, 8, 6)` e treina Backbone + Cabeça inicial."""
        x_grid = flat_to_topographic_grid(x_train)
        self.backbone.trainable = True
        self.head.trainable = True

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.cfg.learning_rate),
            loss="mse",
        )

        cb_list: list[callbacks.Callback] = []
        if self.cfg.validation_split > 0.0 and len(x_grid) > 100:
            cb_list.append(
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.cfg.early_stopping_patience,
                    restore_best_weights=True,
                )
            )

        self.model.fit(
            x_grid,
            y_train,
            epochs=self.cfg.epochs,
            batch_size=self.cfg.batch_size,
            validation_split=self.cfg.validation_split,
            callbacks=cb_list,
            shuffle=True,
            verbose=0,
        )

    def predict(self, x_eval: np.ndarray) -> np.ndarray:
        """Converte `x_eval` para `(N, 8, 8, 6)` e retorna predições `(N, 5)` em `float32`."""
        x_grid = flat_to_topographic_grid(x_eval)
        preds = self.model.predict(x_grid, batch_size=1024, verbose=0)
        return np.asarray(preds, dtype=np.float32)

    def adapt_head(self, x_calib: np.ndarray, y_calib: np.ndarray) -> None:
        """Congela `f_θ` e recalibra apenas a camada linear `regression_head`."""
        x_grid = flat_to_topographic_grid(x_calib)
        self.backbone.trainable = False

        if self.cfg.adaptation_mode == "ridge":
            latent = self.backbone.predict(x_grid, batch_size=1024, verbose=0)
            ridge = Ridge(alpha=self.cfg.ridge_alpha)
            ridge.fit(latent, y_calib)
            w_kernel = ridge.coef_.T.astype(np.float32)
            b_bias = ridge.intercept_.astype(np.float32)
            self.head.set_weights([w_kernel, b_bias])
            return

        if self.cfg.adaptation_mode == "adam":
            self.head.trainable = True
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=self.cfg.calib_lr),
                loss="mse",
            )
            self.model.fit(
                x_grid,
                y_calib,
                epochs=self.cfg.calib_epochs,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                verbose=0,
            )
            return

        raise ValueError(f"Unknown adaptation_mode: {self.cfg.adaptation_mode}")
