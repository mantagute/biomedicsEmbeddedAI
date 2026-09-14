import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from emg_hls4ml_mvp.dataset import build_feature_dataset

def main():
    parser = argparse.ArgumentParser(description="Treinar baseline Ridge para EMG")
    parser.add_argument("--data_root", type=str, default="emg_hls4ml_mvp/data", help="Caminho para a pasta data")
    parser.add_argument("--subject", type=str, default="Sub001", help="Nome do sujeito")
    parser.add_argument("--recording_id", type=str, default="Sub001_1_05_450_0", help="ID da gravacao")
    parser.add_argument("--target", type=str, default="index_z", help="Coluna alvo (cinematica)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Erro: Pasta {data_root} nao encontrada.")
        sys.exit(1)

    print(f"Carregando dados para {args.subject} - {args.recording_id}...")
    X, y, _ = build_feature_dataset(
        data_root=data_root,
        subject=args.subject,
        recording_id=args.recording_id,
        target_column=args.target,
    )

    print(f"Dataset carregado: X shape = {X.shape}, y shape = {y.shape}")

    # Split (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    print("Normalizando as features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Treinando modelo Ridge...")
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import matplotlib.pyplot as plt

    print("Avaliando...")
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nMétricas de Regressão:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R2:   {r2:.4f}")

    print("\nGerando gráficos de avaliação e salvando em 'regression_plots.png'...")
    
    plt.figure(figsize=(12, 5))
    
    # 1. Scatter Plot (Real vs Predito)
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
    
    # Linha ideal (y = x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title("Predito vs Real")
    plt.xlabel("Ângulo Real (Ground Truth)")
    plt.ylabel("Ângulo Previsto")
    
    # 2. Time Series Overlay (Amostra sequencial)
    # Mostrando apenas as primeiras 200 amostras (2 segundos) para ver a curva
    plt.subplot(1, 2, 2)
    n_samples_to_plot = min(200, len(y_test))
    plt.plot(y_test[:n_samples_to_plot], label="Real", color='blue', linewidth=2)
    plt.plot(y_pred[:n_samples_to_plot], label="Previsto", color='orange', linestyle='dashed')
    plt.title(f"Acompanhamento no Tempo (Primeiros {n_samples_to_plot} frames)")
    plt.xlabel("Tempo (frames a 100Hz)")
    plt.ylabel("Ângulo")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("regression_plots.png")
    print("Concluído! Abra o arquivo 'regression_plots.png' para ver como o modelo se saiu.")

if __name__ == "__main__":
    main()
