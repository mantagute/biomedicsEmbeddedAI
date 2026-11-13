"""Camada `controllers/`: Orquestração do protocolo LOSO, barreiras contra Data Leakage e métricas."""

from emg_loso.controllers.loso_controller import (
    FeatureStandardizer,
    FoldEvaluationResult,
    LosoController,
    LosoSummaryResult,
)
from emg_loso.controllers.metrics import (
    FingerMetrics,
    RegressionMetrics,
    compute_regression_metrics,
)

__all__ = [
    "FeatureStandardizer",
    "FingerMetrics",
    "FoldEvaluationResult",
    "LosoController",
    "LosoSummaryResult",
    "RegressionMetrics",
    "compute_regression_metrics",
]
