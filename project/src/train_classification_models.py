import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, learning_curve, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import mlflow
import mlflow.sklearn

from data import load_and_engineer_features, get_preprocessing_pipeline, split_data

warnings.filterwarnings('ignore')

# Random State
RNG = 42

def setup_mlflow(experiment_name="classification_models"):
    mlflow_tracking_uri = os.path.join(os.path.dirname(os.getcwd()), 'mlruns')
    print(f"MLflow tracking URI: {mlflow_tracking_uri}")
    # Change to \\ on Windows
    if os.name == 'nt':
        mlflow.set_tracking_uri(f"file:\\{mlflow_tracking_uri}")
    else:
        mlflow.set_tracking_uri(f"file://{mlflow_tracking_uri}")
    mlflow.end_run()
    mlflow.set_experiment(experiment_name)


def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Below Median', 'Above Median'],
                yticklabels=['Below Median', 'Above Median'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()

    mlflow.log_figure(plt.gcf(), filename)
    plt.close()

    print(f"\nConfusion Matrix ({title}):")
    print(cm)
    print("\nClasses: 0 - Below Median, 1 - Above Median")


def plot_learning_curve(pipeline, X_train, y_train, title, filename, params_str=None):
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X_train, y_train, cv=4,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy', n_jobs=-1, random_state=RNG
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='#2E86AB',
             linewidth=2, markersize=6, label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.2, color='#2E86AB')
    plt.plot(train_sizes, val_mean, 'o-', color='#A23B72',
             linewidth=2, markersize=6, label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.2, color='#A23B72')

    plt.xlabel('Training Set Size', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy Score', fontsize=12, fontweight='bold')

    full_title = f'{title}\n({params_str})' if params_str else title
    plt.title(full_title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    mlflow.log_figure(plt.gcf(), filename)
    plt.close()


def train_naive_bayes(X_train, X_test, y_train, y_test, preprocessor):
    print("\n\n")
    print("TRAINING NAIVE BAYES MODEL")
    print("---\n\n")

    with mlflow.start_run(run_name="Naive_Bayes"):
        mlflow.log_param("model_type", "GaussianNB")
        mlflow.log_param("cv_folds", 4)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", RNG)

        nb_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GaussianNB())
        ])
        nb_pipeline.fit(X_train, y_train)

        cv_scores = cross_val_score(nb_pipeline, X_train, y_train, cv=4, scoring='accuracy')
        mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
        mlflow.log_metric("cv_std_accuracy", cv_scores.std())

        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f}")

        y_train_pred = nb_pipeline.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='binary')

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_f1", train_f1)

        print(f"\nTraining accuracy: {train_accuracy:.4f}")
        print(f"Training F1: {train_f1:.4f}")

        y_test_pred = nb_pipeline.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='binary')

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1", test_f1)

        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test F1: {test_f1:.4f}")

        plot_confusion_matrix(y_test, y_test_pred,
                              'Naive Bayes - Confusion Matrix (Test Set)',
                              'nb_confusion_matrix.png')

        plot_learning_curve(nb_pipeline, X_train, y_train,
                            'Naive Bayes - Learning Curve',
                            'nb_learning_curve.png')

        mlflow.sklearn.log_model(nb_pipeline, "model")

        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_accuracy': train_accuracy,
            'train_f1': train_f1,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1
        }


def train_logistic_regression(X_train, X_test, y_train, y_test, preprocessor):
    print("\n\n")
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("---\n\n")

    with mlflow.start_run(run_name="Logistic_Regression"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("cv_folds", 4)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", RNG)
        mlflow.log_param("scaler", "StandardScaler")

        lr_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=RNG))
        ])

        lr_param_grid = {
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__solver': ['saga', 'liblinear']
        }

        lr_grid_search = GridSearchCV(
            lr_pipeline,
            lr_param_grid,
            cv=4,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        lr_grid_search.fit(X_train, y_train)

        best_params = lr_grid_search.best_params_
        best_pipeline = lr_grid_search.best_estimator_

        print(f"\nBest parameters: {best_params}")
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # Cross-validation with best model
        best_cv_score = lr_grid_search.best_score_
        cv_scores = cross_val_score(best_pipeline, X_train, y_train,
                                    cv=4, scoring='accuracy')

        mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
        mlflow.log_metric("cv_std_accuracy", cv_scores.std())
        mlflow.log_metric("best_cv_score", best_cv_score)

        print(f"Best CV score from grid search: {best_cv_score:.4f}")
        print(f"Cross-validation scores (best model): {cv_scores}")
        print(f"Mean CV accuracy (best model): {cv_scores.mean():.4f}")


        y_train_pred = best_pipeline.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='binary')

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_f1", train_f1)

        print(f"\nTraining accuracy: {train_accuracy:.4f}")
        print(f"Training F1: {train_f1:.4f}")

        y_test_pred = best_pipeline.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='binary')

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1", test_f1)

        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test F1: {test_f1:.4f}")

        plot_confusion_matrix(y_test, y_test_pred,
                              'Logistic Regression - Confusion Matrix (Test Set)',
                              'lr_confusion_matrix.png')

        params_str = f"C={best_params['classifier__C']}, penalty={best_params['classifier__penalty']}, solver={best_params['classifier__solver']}"
        plot_learning_curve(best_pipeline, X_train, y_train,
                            'Logistic Regression - Learning Curve',
                            'lr_learning_curve.png',
                            params_str=params_str)

        mlflow.sklearn.log_model(best_pipeline, "model")

        return {
            'best_params': best_params,
            'best_cv_score': best_cv_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_accuracy': train_accuracy,
            'train_f1': train_f1,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1
        }


def print_summary(nb_metrics, lr_metrics):
    print("\n\n")
    print("CLASSIFICATION METRICS FOR BOTH MODELS")
    print("-" * 40)

    metrics_data = {
        'Model': ['Naive Bayes', 'Logistic Regression'],
        'CV Accuracy': [
            f"{nb_metrics['cv_mean']:.4f}",
            f"{lr_metrics['cv_mean']:.4f}"
        ],
        'Train Accuracy': [
            f"{nb_metrics['train_accuracy']:.4f}",
            f"{lr_metrics['train_accuracy']:.4f}"
        ],
        'Train F1': [
            f"{nb_metrics['train_f1']:.4f}",
            f"{lr_metrics['train_f1']:.4f}"
        ],
        'Test Accuracy': [
            f"{nb_metrics['test_accuracy']:.4f}",
            f"{lr_metrics['test_accuracy']:.4f}"
        ],
        'Test F1': [
            f"{nb_metrics['test_f1']:.4f}",
            f"{lr_metrics['test_f1']:.4f}"
        ],
    }

    metrics_df = pd.DataFrame(metrics_data)
    print(metrics_df.to_string(index=False))
    print("-" * 40)

    print(f"\nLogistic Regression Best Parameters: {lr_metrics['best_params']}")


def main():
    setup_mlflow()

    X, y, median_price = load_and_engineer_features("data/train.csv", random_state=RNG)

    print(f"Using median price threshold: ${median_price:,.2f}")

    preprocessor, numeric_features, categorical_features = get_preprocessing_pipeline(X)

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=RNG)

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    nb_metrics = train_naive_bayes(X_train, X_test, y_train, y_test, preprocessor)

    lr_metrics = train_logistic_regression(X_train, X_test, y_train, y_test, preprocessor)

    print_summary(nb_metrics, lr_metrics)


if __name__ == "__main__":
    main()
