"""
controllers/loso_controller.py
------------------------------
Núcleo do Microkernel (Camada `controllers/`): Orquestrador do Protocolo LOSO.

Responsabilidades:
1. Garantir isolamento estrito contra vazamento de dados via `FoldPartition`.
2. Ajustar a padronização Z-score (`FeatureStandardizer`) exclusivamente nos
   sujeitos de origem (`source_subjects`), nunca tocando no sujeito *held-out*.
3. Executar a sequência oficial da RFC-0001 para qualquer plugin que respeite
   `AdaptableRegressorPort`:
      `fit_backbone(20 sujeitos)` -> `predict Zero-Shot (rep2)` ->
      `adapt_head(rep0 + rep1)`   -> `predict Personalizado (rep2)`
4. Agregar médias e desvios-padrão amostrais (`ddof=1`) entre os N folds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from emg_loso.controllers.metrics import RegressionMetrics, compute_regression_metrics
from emg_loso.domain.entities import FoldPartition
from emg_loso.domain.ports import AdaptableRegressorPort, SubjectCacheRepositoryPort


class FeatureStandardizer:
    """Padronizador Z-score em `float32` para evitar duplicação de RAM em `float64`.

    Guarda `mean_` e `scale_` `(384,)` para posterior exportação direta como
    constantes de pré-processamento no ARM/FPGA.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> FeatureStandardizer:
        self.mean_ = np.mean(x, axis=0, dtype=np.float64).astype(np.float32)
        std = np.std(x, axis=0, dtype=np.float64).astype(np.float32)
        self.scale_ = np.where(std > self.eps, std, np.float32(1.0))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FeatureStandardizer must be fitted before transform().")
        return ((x - self.mean_) / self.scale_).astype(np.float32, copy=False)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


@dataclass(frozen=True, slots=True)
class FoldEvaluationResult:
    """Resultado completo de 1 fold LOSO (Zero-Shot vs. Personalizado)."""

    fold_index: int
    target_subject: str
    model_name: str
    zero_shot: RegressionMetrics
    personalized: RegressionMetrics

    @property
    def r2_gain(self) -> float:
        return self.personalized.mean_r2 - self.zero_shot.mean_r2

    @property
    def rmse_reduction_deg(self) -> float:
        return self.zero_shot.mean_rmse_deg - self.personalized.mean_rmse_deg


@dataclass(frozen=True, slots=True)
class LosoSummaryResult:
    """Agregação estatística entre todos os folds (Mean ± Sample SD, ddof=1)."""

    model_name: str
    folds: tuple[FoldEvaluationResult, ...]

    def aggregate_table(self) -> dict[str, tuple[float, float]]:
        """Retorna `{métrica: (média, desvio_padrão)}` para Zero-Shot e Personalizado."""
        ddof = 1 if len(self.folds) > 1 else 0
        metrics_map = {
            "zs_rmse_deg": [f.zero_shot.mean_rmse_deg for f in self.folds],
            "zs_nrmse_pct": [f.zero_shot.mean_nrmse_pct for f in self.folds],
            "zs_mae_deg": [f.zero_shot.mean_mae_deg for f in self.folds],
            "zs_r2": [f.zero_shot.mean_r2 for f in self.folds],
            "zs_pearson_r": [f.zero_shot.mean_pearson_r for f in self.folds],
            "pers_rmse_deg": [f.personalized.mean_rmse_deg for f in self.folds],
            "pers_nrmse_pct": [f.personalized.mean_nrmse_pct for f in self.folds],
            "pers_mae_deg": [f.personalized.mean_mae_deg for f in self.folds],
            "pers_r2": [f.personalized.mean_r2 for f in self.folds],
            "pers_pearson_r": [f.personalized.mean_pearson_r for f in self.folds],
        }
        return {
            key: (float(np.nanmean(vals)), float(np.nanstd(vals, ddof=ddof)))
            for key, vals in metrics_map.items()
        }

    def format_summary(self) -> str:
        """Formata tabela comparativa Zero-Shot vs. Personalizado pronta para logs/artigo."""
        agg = self.aggregate_table()
        lines = [
            f"=== Resumo LOSO ({len(self.folds)} folds) | Modelo: {self.model_name} ===",
            f"{'Métrica':<16} | {'Zero-Shot':<18} | {'Personalizado':<18}",
            "-" * 58,
        ]
        rows = [
            ("RMSE (°)", "zs_rmse_deg", "pers_rmse_deg"),
            ("NRMSE (%)", "zs_nrmse_pct", "pers_nrmse_pct"),
            ("MAE (°)", "zs_mae_deg", "pers_mae_deg"),
            ("R²", "zs_r2", "pers_r2"),
            ("Pearson r", "zs_pearson_r", "pers_pearson_r"),
        ]
        for label, zs_k, p_k in rows:
            zs_m, zs_s = agg[zs_k]
            p_m, p_s = agg[p_k]
            lines.append(f"{label:<16} | {zs_m:7.3f} ± {zs_s:6.3f}  | {p_m:7.3f} ± {p_s:6.3f}")
        return "\n".join(lines)


class LosoController:
    """Orquestrador principal dos experimentos LOSO e de sanidade intra-sujeito."""

    def __init__(self, cache_repo: SubjectCacheRepositoryPort) -> None:
        self.cache_repo = cache_repo

    def run_within_subject_sanity(
        self,
        subject: str,
        model_factory: Callable[[], AdaptableRegressorPort],
    ) -> RegressionMetrics:
        """Teste rápido em 1 único sujeito: treina em `rep0+rep1` e avalia em `rep2`."""
        ds = self.cache_repo.load_subject(subject)
        x_cal, y_cal = ds.get_calibration_split()
        x_tst, y_tst = ds.get_test_split()

        scaler = FeatureStandardizer()
        x_cal_scaled = scaler.fit_transform(x_cal)
        x_tst_scaled = scaler.transform(x_tst)

        model = model_factory()
        model.fit_backbone(x_cal_scaled, y_cal)
        y_pred = model.predict(x_tst_scaled)
        return compute_regression_metrics(y_tst, y_pred)

    def run_single_fold(
        self,
        partition: FoldPartition,
        model_factory: Callable[[], AdaptableRegressorPort],
    ) -> FoldEvaluationResult:
        """Executa 1 fold completo do protocolo LOSO (Zero-Shot -> Head-TL)."""
        # 1. Carregar e concatenar os sujeitos de origem (treino base)
        x_src_list: list[np.ndarray] = []
        y_src_list: list[np.ndarray] = []
        for src_sub in partition.source_subjects:
            src_ds = self.cache_repo.load_subject(src_sub)
            x_src_list.append(src_ds.x_flat)
            y_src_list.append(src_ds.y_angles)

        x_train = np.concatenate(x_src_list, axis=0)
        y_train = np.concatenate(y_src_list, axis=0)

        # 2. Carregar o sujeito held-out separado em calibração (rep0+1) e teste (rep2)
        target_ds = self.cache_repo.load_subject(partition.target_subject)
        x_calib, y_calib = target_ds.get_calibration_split()
        x_test, y_test = target_ds.get_test_split()

        # 3. Barreira de normalização: StandardScaler ajustado APENAS nos sujeitos de treino
        scaler = FeatureStandardizer()
        x_train_scaled = scaler.fit_transform(x_train)
        x_calib_scaled = scaler.transform(x_calib)
        x_test_scaled = scaler.transform(x_test)

        # 4. Treino do Backbone generalista (Fábrica)
        model = model_factory()
        model.fit_backbone(x_train_scaled, y_train)

        # 5. Avaliação A: Zero-Shot (sem olhar para nenhum dado do sujeito alvo)
        y_pred_zs = model.predict(x_test_scaled)
        zs_metrics = compute_regression_metrics(y_test, y_pred_zs)

        # 6. Adaptação da Cabeça Linear (Personalização usando apenas rep0 + rep1)
        model.adapt_head(x_calib_scaled, y_calib)

        # 7. Avaliação B: Personalizado (no mesmo rep2 intocado)
        y_pred_pers = model.predict(x_test_scaled)
        pers_metrics = compute_regression_metrics(y_test, y_pred_pers)

        return FoldEvaluationResult(
            fold_index=partition.fold_index,
            target_subject=partition.target_subject,
            model_name=model.name,
            zero_shot=zs_metrics,
            personalized=pers_metrics,
        )

    def run_loso(
        self,
        subjects: Sequence[str],
        model_factory: Callable[[], AdaptableRegressorPort],
        verbose: bool = True,
    ) -> LosoSummaryResult:
        """Executa os K folds do protocolo LOSO sobre a lista de sujeitos informada."""
        if len(subjects) < 2:
            raise ValueError("LOSO requires at least 2 subjects.")

        results: list[FoldEvaluationResult] = []
        for idx, target in enumerate(subjects):
            sources = tuple(s for s in subjects if s != target)
            partition = FoldPartition(
                fold_index=idx,
                target_subject=target,
                source_subjects=sources,
            )
            fold_res = self.run_single_fold(partition, model_factory=model_factory)
            results.append(fold_res)

            if verbose:
                print(
                    f"[Fold {idx + 1:02d}/{len(subjects):02d} | {target}] "
                    f"ZS R²={fold_res.zero_shot.mean_r2:+.3f} (RMSE={fold_res.zero_shot.mean_rmse_deg:.2f}°) -> "
                    f"Pers R²={fold_res.personalized.mean_r2:+.3f} (RMSE={fold_res.personalized.mean_rmse_deg:.2f}°)"
                )

        summary = LosoSummaryResult(
            model_name=results[0].model_name,
            folds=tuple(results),
        )
        if verbose:
            print("\n" + summary.format_summary())
        return summary
