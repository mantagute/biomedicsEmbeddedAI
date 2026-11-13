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


class _TopographicBatchSequence(tf.keras.utils.Sequence):
    """Gerador de lotes sob demanda `(B, 384) -> (B, 8, 8, 6)` para evitar cópia de 5+ GB em RAM."""

    def __init__(
        self,
        x_flat: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.x_flat = x_flat
        self.y = y
        self.indices = indices.copy()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        start = idx * self.batch_size
        batch_idx = self.indices[start : start + self.batch_size]
        x_batch = flat_to_topographic_grid(self.x_flat[batch_idx])
        return x_batch, self.y[batch_idx]

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)


class TopographicCNN2D:
    """Implementação do contrato `AdaptableRegressorPort` usando CNN 2D topográfica."""

    def __init__(self, cfg: Cnn2dConfig = Cnn2dConfig()) -> None:
        self.cfg = cfg
        tf.keras.backend.clear_session()
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
        """Treina Backbone + Cabeça inicial gerando tensores `(B, 8, 8, 6)` por lote."""
        self.backbone.trainable = True
        self.head.trainable = True

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.cfg.learning_rate),
            loss="mse",
        )

        n_samples = len(x_train)
        all_idx = np.arange(n_samples)
        cb_list: list[callbacks.Callback] = []
        val_seq = None

        if self.cfg.validation_split > 0.0 and n_samples > 100:
            rng = np.random.default_rng(self.cfg.seed)
            perm = rng.permutation(n_samples)
            n_val = max(1, int(n_samples * self.cfg.validation_split))
            val_idx, train_idx = perm[:n_val], perm[n_val:]
            val_seq = _TopographicBatchSequence(
                x_train, y_train, val_idx, batch_size=1024, shuffle=False, seed=self.cfg.seed
            )
            cb_list.append(
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.cfg.early_stopping_patience,
                    restore_best_weights=True,
                )
            )
        else:
            train_idx = all_idx

        train_seq = _TopographicBatchSequence(
            x_train,
            y_train,
            train_idx,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            seed=self.cfg.seed,
        )

        self.model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=self.cfg.epochs,
            callbacks=cb_list,
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
