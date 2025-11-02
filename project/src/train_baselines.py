from train_regression_models import main as train_regression_models
from train_classification_models import main as train_classification_models

def print_regression_summary(metrics):
    print("\n\n")
    print("REGRESSION METRICS FOR BOTH MODELS")
    print("-" * 80)

    header = f"{'Model':<25} {'CV RMSE':>12} {'Train RMSE':>12} {'Train R²':>10} {'Test RMSE':>12} {'Test R²':>10}"
    print(header)

    ridge = metrics['ridge']
    ridge_row = f"{'Ridge':<25} ${ridge['cv_rmse']:>10,.2f} ${ridge['train_rmse']:>10,.2f} {ridge['train_r2']:>10.4f} ${ridge['test_rmse']:>10,.2f} {ridge['test_r2']:>10.4f}"
    print(ridge_row)

    dt = metrics['decision_tree']
    dt_row = f"{'Decision Tree':<25} ${dt['cv_rmse']:>10,.2f} ${dt['train_rmse']:>10,.2f} {dt['train_r2']:>10.4f} ${dt['test_rmse']:>10,.2f} {dt['test_r2']:>10.4f}"
    print(dt_row)

    print("-" * 80)

def print_classification_summary(metrics):
    print("\n\n")
    print("CLASSIFICATION METRICS FOR BOTH MODELS")
    print("-" * 80)

    header = f"{'Model':<25} {'CV Accuracy':>12} {'Train Accuracy':>15} {'Train F1':>10} {'Test Accuracy':>14} {'Test F1':>10}"
    print(header)

    nb = metrics['naive_bayes']
    nb_row = f"{'Naive Bayes':<25} {nb['cv_mean']:>12.4f} {nb['train_accuracy']:>15.4f} {nb['train_f1']:>10.4f} {nb['test_accuracy']:>14.4f} {nb['test_f1']:>10.4f}"
    print(nb_row)

    lr = metrics['logistic_regression']
    lr_row = f"{'Logistic Regression':<25} {lr['cv_mean']:>12.4f} {lr['train_accuracy']:>15.4f} {lr['train_f1']:>10.4f} {lr['test_accuracy']:>14.4f} {lr['test_f1']:>10.4f}"
    print(lr_row)

    print("-" * 80)

def main():
    regression_metrics = train_regression_models()
    print("\n" + "-"*50 + "\n")
    classification_metrics = train_classification_models()

    print("\n" + "="*80)
    print("FINAL SUMMARY - ALL MODELS")
    print("="*80)

    print_regression_summary(regression_metrics)
    print_classification_summary(classification_metrics)

if __name__ == "__main__":
    main()