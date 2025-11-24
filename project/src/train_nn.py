from sklearn.metrics import classification_report

from train_regression_nn import train_regression_nn

from train_classification_nn import train_classification_nn


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

    classification_metrics = train_classification_nn()

    print("\n" + "-"*40)
    print("CLASSIFICATION NN - FINAL RESULTS")
    print("-"*40)
    print("\nTraining Set Performance:")
    print(f"Accuracy: {classification_metrics['train_accuracy']:.2f}")
    print(f"F1: {classification_metrics['train_f1']:.2f}")
    print("\nTest Set Performance:")
    print(f"Accuracy: {classification_metrics['test_accuracy']:.2f}")
    print(f"F1: {classification_metrics['test_f1']:.2f}")
    print("-"*40)


if __name__ == "__main__":
    main()
