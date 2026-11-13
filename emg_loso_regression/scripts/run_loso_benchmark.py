"""
scripts/run_loso_benchmark.py
-----------------------------
Executor oficial do protocolo LOSO (Leave-One-Subject-Out) para o artigo.

Executa os modelos plugados (`ridge`, `mlp`, `cnn2d`) sobre os sujeitos em cache,
registrando métricas Zero-Shot vs. Head Personalization por fold, por dedo e
agregadas (Média ± Desvio Padrão), e exportando CSVs/JSON para análise e gráficos.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
from sklearn.linear_model import Ridge

from emg_loso.connectors import (
    Cnn2dConfig,
    DeepMLPRegressor,
    MlpConfig,
    TopographicCNN2D,
)
from emg_loso.controllers import LosoController
from emg_loso.controllers.loso_controller import LosoSummaryResult
from emg_loso.domain.ports import AdaptableRegressorPort
from emg_loso.repositories import NpzSubjectCacheRepository


class _RidgeBaseline:
    """Baseline linear (Ridge) implementando `AdaptableRegressorPort`."""

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self._model = Ridge(alpha=alpha)

    @property
    def name(self) -> str:
        return f"RidgeBaseline(alpha={self.alpha})"

    def fit_backbone(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(x_train, y_train)

    def predict(self, x_eval: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(x_eval), dtype=np.float32)

    def adapt_head(self, x_calib: np.ndarray, y_calib: np.ndarray) -> None:
        self._model.fit(x_calib, y_calib)


def build_model_factory(
    model_key: str,
    epochs: int,
    adaptation_mode: str,
    ridge_alpha: float,
) -> Callable[[], AdaptableRegressorPort]:
    """Cria a factory de modelo correspondente à chave CLI."""
    if model_key == "ridge":
        return lambda: _RidgeBaseline(alpha=ridge_alpha)
    if model_key == "mlp":
        cfg = MlpConfig(
            epochs=epochs,
            adaptation_mode=adaptation_mode,  # type: ignore[arg-type]
            ridge_alpha=ridge_alpha,
        )
        return lambda: DeepMLPRegressor(cfg)
    if model_key == "cnn2d":
        cfg_cnn = Cnn2dConfig(
            epochs=epochs,
            adaptation_mode=adaptation_mode,  # type: ignore[arg-type]
            ridge_alpha=ridge_alpha,
        )
        return lambda: TopographicCNN2D(cfg_cnn)
    raise ValueError(f"Modelo desconhecido: {model_key}")


def export_results(
    summaries: list[LosoSummaryResult],
    out_dir: Path,
) -> None:
    """Exporta resultados por fold, por dedo e resumo agregado para CSV/JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)

    folds_csv = out_dir / "loso_folds.csv"
    with folds_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "fold_index",
            "target_subject",
            "regime",
            "mean_rmse_deg",
            "mean_mae_deg",
            "mean_nrmse_pct",
            "mean_r2",
            "mean_pearson_r",
        ])
        for s in summaries:
            for fold in s.folds:
                for regime, m in (
                    ("zero_shot", fold.zero_shot),
                    ("personalized", fold.personalized),
                ):
                    writer.writerow([
                        s.model_name,
                        fold.fold_index,
                        fold.target_subject,
                        regime,
                        f"{m.mean_rmse_deg:.4f}",
                        f"{m.mean_mae_deg:.4f}",
                        f"{m.mean_nrmse_pct:.4f}",
                        f"{m.mean_r2:.6f}",
                        f"{m.mean_pearson_r:.6f}",
                    ])

    fingers_csv = out_dir / "loso_per_finger.csv"
    with fingers_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "fold_index",
            "target_subject",
            "regime",
            "dof",
            "rmse_deg",
            "mae_deg",
            "nrmse_pct",
            "r2",
            "pearson_r",
        ])
        for s in summaries:
            for fold in s.folds:
                for regime, m in (
                    ("zero_shot", fold.zero_shot),
                    ("personalized", fold.personalized),
                ):
                    for finger in m.per_finger:
                        writer.writerow([
                            s.model_name,
                            fold.fold_index,
                            fold.target_subject,
                            regime,
                            finger.dof,
                            f"{finger.rmse_deg:.4f}",
                            f"{finger.mae_deg:.4f}",
                            f"{finger.nrmse_pct:.4f}",
                            f"{finger.r2:.6f}",
                            f"{finger.pearson_r:.6f}",
                        ])

    summary_json = out_dir / "loso_summary.json"
    payload = [
        {
            "model": s.model_name,
            "n_folds": len(s.folds),
            "aggregate": {
                k: {"mean": mean_val, "sd": sd_val}
                for k, (mean_val, sd_val) in s.aggregate_table().items()
            },
        }
        for s in summaries
    ]
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nResultados exportados em:\n  - {folds_csv}\n  - {fingers_csv}\n  - {summary_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runner LOSO Multi-DoF HD-sEMG")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "cache_hw_causal",
        help="Diretório contendo os caches SubXXX.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
        help="Diretório de saída para CSVs e JSON de resultados",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "mlp", "cnn2d"],
        choices=["ridge", "mlp", "cnn2d"],
        help="Modelos a avaliar no protocolo LOSO",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Subconjunto opcional de sujeitos (ex: Sub001 Sub002 Sub003). Padrão: todos em cache.",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Épocas de treino do backbone")
    parser.add_argument(
        "--adaptation-mode",
        choices=["ridge", "adam"],
        default="ridge",
        help="Modo de personalização da cabeça linear",
    )
    parser.add_argument("--ridge-alpha", type=float, default=10.0, help="Regularização Ridge")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = NpzSubjectCacheRepository(args.cache_dir)
    subjects = args.subjects if args.subjects is not None else repo.list_cached_subjects()
    if len(subjects) < 2:
        raise SystemExit(f"São necessários pelo menos 2 sujeitos em cache; encontrados: {subjects}")

    print(f"Executando LOSO em {len(subjects)} sujeitos: {subjects}")
    ctrl = LosoController(repo)
    summaries: list[LosoSummaryResult] = []

    for model_key in args.models:
        t0 = time.perf_counter()
        print(f"\n{'=' * 70}\nModelo: {model_key}\n{'=' * 70}")
        factory = build_model_factory(
            model_key=model_key,
            epochs=args.epochs,
            adaptation_mode=args.adaptation_mode,
            ridge_alpha=args.ridge_alpha,
        )
        summary = ctrl.run_loso(subjects=subjects, model_factory=factory, verbose=True)
        summaries.append(summary)
        print(f"Tempo total ({model_key}): {time.perf_counter() - t0:.1f}s")

    export_results(summaries, args.out_dir)


if __name__ == "__main__":
    main()
