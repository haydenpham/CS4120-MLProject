from train_regression_nn import train_regression_nn


def main():
    regression_metrics = train_regression_nn()

    print("\n" + "-"*40)
    print("REGRESSION NN - FINAL RESULTS")
    print("-"*40)
    print("\nTraining Set Performance:")
    print(f"RMSE: {regression_metrics['train_rmse']:.2f}")
    print(f"MAE: {regression_metrics['train_mae']:.2f}")
    print(f"R²: {regression_metrics['train_r2']:.4f}")
    print("\nTest Set Performance:")
    print(f"RMSE: {regression_metrics['test_rmse']:.2f}")
    print(f"MAE: {regression_metrics['test_mae']:.2f}")
    print(f"R²: {regression_metrics['test_r2']:.4f}")
    print("-"*40)


if __name__ == "__main__":
    main()