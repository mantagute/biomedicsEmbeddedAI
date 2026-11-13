"""Contratos internos (Protocols) para desacoplar o controller dos repositories e connectors."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from emg_loso.domain.entities import SubjectDataset


class SubjectCacheRepositoryPort(Protocol):
    """Contrato do repository responsável por persistir e carregar o cache `.npz` por sujeito."""

    def exists(self, subject: str) -> bool:
        """Retorna True se o cache do sujeito já existir em disco."""
        ...

    def save_subject(self, dataset: SubjectDataset) -> Path:
        """Persiste o `SubjectDataset` em disco e retorna o caminho do arquivo salvo."""
        ...

    def load_subject(self, subject: str) -> SubjectDataset:
        """Carrega o `SubjectDataset` validado a partir do cache em disco."""
        ...

    def list_cached_subjects(self) -> list[str]:
        """Lista os identificadores (`SubXXX`) disponíveis no diretório de cache."""
        ...


class AdaptableRegressorPort(Protocol):
    """Contrato dos plugins de Deep Learning (Microkernel) compatíveis com o LOSOController.

    Qualquer arquitetura (`DeepMLPRegressor`, `TopographicCNN2D`, `TCN`) deve expor
    exatamente estas três fases para garantir comparação científica justa:
    1. `fit_backbone`: treina extrator + cabeça nos 20 sujeitos de origem.
    2. `predict`: realiza inferência contínua retornando shape `(N, 5)`.
    3. `adapt_head`: congela o extrator `f_θ` e recalibra apenas a cabeça linear
       usando `rep0 + rep1` do sujeito alvo.
    """

    @property
    def name(self) -> str:
        """Identificador legível do modelo para logs, tabelas e gráficos."""
        ...

    def fit_backbone(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Treina o extrator de características e a cabeça inicial nos sujeitos de origem."""
        ...

    def predict(self, x_eval: np.ndarray) -> np.ndarray:
        """Prediz os ângulos contínuos `(N, 5)` sem alterar pesos do modelo."""
        ...

    def adapt_head(self, x_calib: np.ndarray, y_calib: np.ndarray) -> None:
        """Congela `f_θ` e adapta apenas a camada linear final com `rep0 + rep1`."""
        ...
