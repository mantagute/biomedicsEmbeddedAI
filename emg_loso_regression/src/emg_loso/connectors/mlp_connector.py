"""
connectors/mlp_connector.py
---------------------------
Plugin 1 do Microkernel (Camada `connectors/`): Deep MLP Regressor em Keras.

Arquitetura Hardware-Aware (`hls4ml`-compatível):
- Input: vetor plano `(384,)` (`[RMS | MAV | WL]`)
- Backbone `f_θ`: `Dense(64) -> BatchNorm -> ReLU -> Dropout -> Dense(32) -> BatchNorm -> ReLU`
- Regression Head: `Dense(5, linear)` prevendo os 5 dedos simultaneamente.

Destaque de Engenharia (Head Personalization):
Suporta dois modos de adaptação da cabeça linear usando `rep0 + rep1`:
1. `adaptation_mode="ridge"` (Padrão): Congela `f_θ`, extrai as ativações latentes
   `H = f_θ(X_calib)` e resolve a regressão linear regularizada em forma fechada,
   injetando os pesos exatos `(W, b)` de volta na camada `Dense(5)` do Keras.
   Isso garante ótimo global convexo em milissegundos e mantém o modelo 100%
   exportável para o `hls4ml`.
2. `adaptation_mode="adam"`: Congela os pesos do Backbone e faz fine-tuning por
   gradiente apenas na camada `Dense(5)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.linear_model import Ridge
import tensorflow as tf
from tensorflow.keras import Model, callbacks, layers, optimizers

from emg_loso.config import N_DOFS
from emg_loso.domain.signal.topography import EXPECTED_FLAT_DIM


@dataclass(frozen=True, slots=True)
class MlpConfig:
    """Hiperparâmetros do plugin `DeepMLPRegressor`."""

    hidden_units: tuple[int, ...] = (64, 32)
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


class DeepMLPRegressor:
    """Implementação do contrato `AdaptableRegressorPort` usando MLP em Keras."""

    def __init__(self, cfg: MlpConfig = MlpConfig()) -> None:
        self.cfg = cfg
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(cfg.seed)
        self.model, self.backbone, self.head = self._build_model()

    @property
    def name(self) -> str:
        dims = "-".join(str(u) for u in self.cfg.hidden_units)
        return f"DeepMLP({dims}, adapt={self.cfg.adaptation_mode})"

    def _build_model(self) -> tuple[Model, Model, layers.Dense]:
        inputs = layers.Input(shape=(EXPECTED_FLAT_DIM,), name="emg_features_in")
        x = inputs

        for idx, units in enumerate(self.cfg.hidden_units, start=1):
            x = layers.Dense(units, use_bias=False, name=f"fc{idx}")(x)
            x = layers.BatchNormalization(name=f"bn{idx}")(x)
            x = layers.Activation("relu", name=f"relu{idx}")(x)
            if self.cfg.dropout_rate > 0.0 and idx < len(self.cfg.hidden_units):
                x = layers.Dropout(self.cfg.dropout_rate, name=f"drop{idx}")(x)

        backbone = Model(inputs=inputs, outputs=x, name="mlp_backbone")
        head = layers.Dense(N_DOFS, activation="linear", name="regression_head")
        outputs = head(x)
        full_model = Model(inputs=inputs, outputs=outputs, name="deep_mlp_regressor")
        return full_model, backbone, head

    def fit_backbone(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Treina o Backbone + Cabeça inicial nos dados de origem."""
        self.backbone.trainable = True
        self.head.trainable = True

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.cfg.learning_rate),
            loss="mse",
        )

        cb_list: list[callbacks.Callback] = []
        if self.cfg.validation_split > 0.0 and len(x_train) > 100:
            cb_list.append(
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.cfg.early_stopping_patience,
                    restore_best_weights=True,
                )
            )

        self.model.fit(
            x_train,
            y_train,
            epochs=self.cfg.epochs,
            batch_size=self.cfg.batch_size,
            validation_split=self.cfg.validation_split,
            callbacks=cb_list,
            shuffle=True,
            verbose=0,
        )

    def predict(self, x_eval: np.ndarray) -> np.ndarray:
        """Executa inferência em lote retornando predições `(N, 5)` em `float32`."""
        preds = self.model.predict(x_eval, batch_size=1024, verbose=0)
        return np.asarray(preds, dtype=np.float32)

    def adapt_head(self, x_calib: np.ndarray, y_calib: np.ndarray) -> None:
        """Congela `f_θ` e recalibra apenas a camada linear `regression_head`."""
        self.backbone.trainable = False

        if self.cfg.adaptation_mode == "ridge":
            latent = self.backbone.predict(x_calib, batch_size=1024, verbose=0)
            ridge = Ridge(alpha=self.cfg.ridge_alpha)
            ridge.fit(latent, y_calib)
            # Keras Dense espera kernel com shape (latent_dim, n_dofs) e bias (n_dofs,)
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
                x_calib,
                y_calib,
                epochs=self.cfg.calib_epochs,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                verbose=0,
            )
            return

        raise ValueError(f"Unknown adaptation_mode: {self.cfg.adaptation_mode}")
