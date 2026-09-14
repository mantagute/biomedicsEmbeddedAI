"""
run_baseline.py
---------------
Script de linha de comando para treinar e avaliar o baseline Ridge.

Uso:
    python -m emg_hls4ml_mvp.run_baseline
    python -m emg_hls4ml_mvp.run_baseline --data_root path/to/data --subject Sub002

Este script é equivalente ao notebook 02_baseline_regression.ipynb,
mas em formato executável — útil para rodar em servidores sem interface gráfica.
A saída de avaliação é salva em 'ridge_evaluation.png'.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from emg_hls4ml_mvp.dataset import build_feature_dataset


# ---------------------------------------------------------------------------
# Funções de avaliação
# ---------------------------------------------------------------------------

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula MSE, RMSE, MAE e R² e imprime no terminal."""
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

    print("\nMétricas de Regressão:")
    for name, value in metrics.items():
        print(f"  {name:<5}: {value:.4f}")

    return metrics


def plot_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = "ridge_evaluation.png",
    n_time_samples: int = 200,
) -> None:
    """Gera dois gráficos de avaliação e salva em disco.

    Gráfico 1 — Dispersão (Predito vs Real):
        Um ponto por amostra de teste. Numa predição perfeita, todos os pontos
        ficam sobre a diagonal vermelha (y = x).

    Gráfico 2 — Série Temporal:
        Sobrepõe a curva real (azul) com a predita (laranja) nos primeiros
        `n_time_samples` frames de teste (~2 segundos a 100 Hz).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Gráfico 1: Dispersão ---
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.3, color="steelblue", s=8)

    # Linha ideal y = x
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="ideal")

    ax.set_xlabel("Ângulo Real (ground truth)")
    ax.set_ylabel("Ângulo Previsto")
    ax.set_title("Predito vs Real")
    ax.legend()

    # --- Gráfico 2: Série Temporal ---
    ax = axes[1]
    n = min(n_time_samples, len(y_true))

    ax.plot(y_true[:n],  label="Real",     color="steelblue", linewidth=2)
    ax.plot(y_pred[:n],  label="Previsto", color="darkorange", linestyle="dashed")

    ax.set_xlabel(f"Tempo (frames a 100 Hz)")
    ax.set_ylabel("Ângulo")
    ax.set_title(f"Acompanhamento no Tempo (primeiros {n} frames)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nGráfico salvo em '{output_path}'")


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treinar e avaliar o baseline Ridge para regressão de cinemática de dedos."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Caminho para a pasta data/ (padrão: data/)",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="Sub001",
        help="Identificador do sujeito (padrão: Sub001)",
    )
    parser.add_argument(
        "--recording_id",
        type=str,
        default="Sub001_1_05_450_0",
        help="ID da gravação (padrão: Sub001_1_05_450_0)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="index_z",
        help="Coluna alvo do CSV de ângulos (padrão: index_z)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ridge_evaluation.png",
        help="Caminho do arquivo de saída dos gráficos (padrão: ridge_evaluation.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Erro: pasta '{data_root}' não encontrada.")
        sys.exit(1)

    # --- Etapa 1: Carregar e preparar os dados ---
    print(f"Sujeito: {args.subject} | Gravação: {args.recording_id} | Alvo: {args.target}")
    X, y, _ = build_feature_dataset(
        data_root=data_root,
        subject=args.subject,
        recording_id=args.recording_id,
        target_column=args.target,
        verbose=True,
    )

    # Split cronológico: os últimos 20% do tempo formam o conjunto de teste.
    # shuffle=False é obrigatório para séries temporais — embaralhar causaria
    # vazamento de informação (data leakage) do futuro para o treino.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # --- Etapa 2: Normalização ---
    # O scaler é ajustado apenas no treino e aplicado ao teste,
    # simulando um cenário de deploy onde os dados de teste são desconhecidos.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # --- Etapa 3: Treino do modelo Ridge ---
    # Ridge é regressão linear com regularização L2.
    # Alpha=1.0 é o ponto de partida padrão; aumentar reduz overfitting em datasets maiores.
    print("\nTreinando Ridge Regression (alpha=1.0)...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)

    # --- Etapa 4: Avaliação e visualização ---
    y_pred = ridge.predict(X_test_scaled)
    evaluate_regression(y_test, y_pred)
    plot_evaluation(y_test, y_pred, output_path=args.output)


if __name__ == "__main__":
    main()
