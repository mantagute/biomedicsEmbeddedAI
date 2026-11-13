"""
controllers/metrics.py
----------------------
Cálculo Hand-Made Vetorizado dos Parâmetros de Análise de Regressão Contínua.

Por que Hand-Made em NumPy puro em vez de chamar `sklearn.metrics` e `scipy.stats` em loop?
1. Paralelismo Multi-DoF em 1 chamada: calcula RMSE, MAE, NRMSE (%), R² e
   Correlação de Pearson (r) para os 5 dedos simultaneamente via `axis=0`,
   sem loops Python por dedo ou por gravação.
2. Proteção contra Variância Nula (Dedos em Repouso): em tarefas isoladas
   (ex: Tarefa 1 - Index), dedos não-ativos podem ter excursão angular próxima de
   zero (`SS_tot ≈ 0`). O cálculo explícito com máscara `eps` evita divisões
   por zero silenciosas e agrega médias globais com `np.nanmean`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from emg_loso.config import FINGER_DOF_COLUMNS, N_DOFS
from emg_loso.domain.exceptions import InvalidSignalShapeError

_EPS: Final[float] = 1e-8


@dataclass(frozen=True, slots=True)
class FingerMetrics:
    """Métricas de regressão contínua para um Grau de Liberdade (dedo) individual."""

    dof: str
    rmse_deg: float
    mae_deg: float
    nrmse_pct: float
    r2: float
    pearson_r: float


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    """Pacote completo de métricas Multi-DoF (por dedo + média macro)."""

    per_finger: tuple[FingerMetrics, ...]
    mean_rmse_deg: float
    mean_mae_deg: float
    mean_nrmse_pct: float
    mean_r2: float
    mean_pearson_r: float

    def to_dict(self) -> dict[str, float]:
        """Serializa as médias globais e por dedo para exportação em CSV/tabelas."""
        out: dict[str, float] = {
            "mean_rmse_deg": self.mean_rmse_deg,
            "mean_mae_deg": self.mean_mae_deg,
            "mean_nrmse_pct": self.mean_nrmse_pct,
            "mean_r2": self.mean_r2,
            "mean_pearson_r": self.mean_pearson_r,
        }
        for fm in self.per_finger:
            out[f"{fm.dof}_rmse_deg"] = fm.rmse_deg
            out[f"{fm.dof}_nrmse_pct"] = fm.nrmse_pct
            out[f"{fm.dof}_r2"] = fm.r2
            out[f"{fm.dof}_pearson_r"] = fm.pearson_r
        return out


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dof_names: tuple[str, ...] = FINGER_DOF_COLUMNS,
) -> RegressionMetrics:
    """Calcula RMSE (°), MAE (°), NRMSE (%), R² e Pearson r por dedo e média macro.

    Parâmetros
    ----------
    y_true : np.ndarray, shape (n_samples, n_dofs)
        Ângulos reais medidos pela câmera Vicon (em graus).
    y_pred : np.ndarray, shape (n_samples, n_dofs)
        Ângulos preditos pelo modelo (em graus).
    dof_names : tuple[str, ...]
        Nomes dos DoFs correspondentes às colunas.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    if y_t.ndim == 1:
        y_t = y_t[:, None]
    if y_p.ndim == 1:
        y_p = y_p[:, None]

    if y_t.shape != y_p.shape:
        raise InvalidSignalShapeError(
            f"y_true shape {y_t.shape} does not match y_pred shape {y_p.shape}."
        )
    if y_t.shape[1] != len(dof_names):
        raise InvalidSignalShapeError(
            f"Expected {len(dof_names)} DoF columns ({dof_names}), got {y_t.shape[1]}."
        )
    if y_t.shape[0] < 2:
        raise InvalidSignalShapeError(
            f"Need at least 2 samples to compute R² and Pearson r (got {y_t.shape[0]})."
        )

    errors = y_t - y_p
    mse = np.mean(np.square(errors), axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors), axis=0)

    # NRMSE (%) normalizado pela excursão angular real (max - min)
    y_range = np.ptp(y_t, axis=0)
    nrmse_pct = np.where(y_range > _EPS, 100.0 * rmse / y_range, np.nan)

    # Coeficiente de determinação R² = 1 - SS_res / SS_tot
    y_t_centered = y_t - np.mean(y_t, axis=0, keepdims=True)
    ss_res = np.sum(np.square(errors), axis=0)
    ss_tot = np.sum(np.square(y_t_centered), axis=0)
    r2 = np.where(ss_tot > _EPS, 1.0 - (ss_res / ss_tot), np.nan)

    # Correlação linear de Pearson r = cov(y_t, y_p) / (std(y_t) * std(y_p))
    y_p_centered = y_p - np.mean(y_p, axis=0, keepdims=True)
    cov_num = np.sum(y_t_centered * y_p_centered, axis=0)
    denom = np.sqrt(ss_tot * np.sum(np.square(y_p_centered), axis=0))
    pearson_r = np.where(denom > _EPS, cov_num / denom, np.nan)

    per_finger = tuple(
        FingerMetrics(
            dof=dof_names[i],
            rmse_deg=float(rmse[i]),
            mae_deg=float(mae[i]),
            nrmse_pct=float(nrmse_pct[i]),
            r2=float(r2[i]),
            pearson_r=float(pearson_r[i]),
        )
        for i in range(len(dof_names))
    )

    return RegressionMetrics(
        per_finger=per_finger,
        mean_rmse_deg=float(np.nanmean(rmse)),
        mean_mae_deg=float(np.nanmean(mae)),
        mean_nrmse_pct=float(np.nanmean(nrmse_pct)),
        mean_r2=float(np.nanmean(r2)),
        mean_pearson_r=float(np.nanmean(pearson_r)),
    )
