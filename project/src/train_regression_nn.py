import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import warnings
import os

warnings.filterwarnings('ignore')


def load_and_engineer_regression_features(data_path):
    df = pd.read_csv(data_path)

    df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    df = df.drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], axis=1)

    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    df = df.drop(['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'], axis=1)

    df = df.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating'], axis=1)

    df = df.drop(['MoSold', 'YrSold'], axis=1)

    df['OverallScore'] = df['OverallQual'] * df['OverallCond']

    df = df.drop(['GarageType', 'GarageFinish', 'Fence', 'Alley'], axis=1)
    df = df.drop(['LowQualFinSF', 'PoolArea', 'MiscVal'], axis=1)
    df = df.drop(['GarageQual', 'GarageCond'], axis=1)
    df = df.drop(['PoolQC', 'MiscFeature'], axis=1)

    X = df.drop(['Id', 'SalePrice'], axis=1)
    y = df['SalePrice']

    return X, y


def get_regression_preprocessor(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train['MSSubClass'] = X_train['MSSubClass'].astype(str)
    X_test['MSSubClass'] = X_test['MSSubClass'].astype(str)

    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    return preprocessor, X_train, X_test


def train_regression_nn(random_seed=42):
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
    mlflow_tracking_uri = os.path.join(os.path.dirname(__file__), '..', 'mlruns')
    figures_dir = os.path.join(os.path.dirname(__file__), '..', 'figures_final')

    os.makedirs(figures_dir, exist_ok=True)

    if os.name == 'nt':
        mlflow.set_tracking_uri(f"file:\\{mlflow_tracking_uri}")
    else:
        mlflow.set_tracking_uri(f"file://{mlflow_tracking_uri}")

    mlflow.end_run()
    mlflow.set_experiment("regression_nn")

    X, y = load_and_engineer_regression_features(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed
    )

    preprocessor, X_train, X_test = get_regression_preprocessor(X_train, X_test)

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    mlp_for_grid = MLPRegressor(
        max_iter=500,
        random_state=random_seed,
        early_stopping=True,
        n_iter_no_change=20,
        verbose=False,
    )

    param_grid = {
        'hidden_layer_sizes': [
            (100,),
            (100, 50),
            (64, 32),
            (128, 64, 32),
        ],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001, 0.01]
    }

    grid_search = GridSearchCV(
        mlp_for_grid,
        param_grid,
        cv=4,
        scoring='neg_mean_squared_error',
    )

    grid_search.fit(X_train_preprocessed, y_train)
    best_params = grid_search.best_params_

    with mlflow.start_run(run_name="regression_nn_final_model"):
        mlflow.log_params(best_params)
        mlflow.log_param("final_max_iter", 100)
        mlflow.log_param("random_seed", random_seed)

        X_t, X_v, y_t, y_v = train_test_split(
            X_train_preprocessed, y_train, test_size=0.2, random_state=random_seed
        )

        model = MLPRegressor(
            **best_params,
            max_iter=1,
            warm_start=True,
            random_state=random_seed
        )

        train_loss_history = []
        val_loss_history = []

        epochs = 200

        for epoch in range(epochs):
            model.fit(X_t, y_t)
            train_loss_history.append(model.loss_)

            val_pred = model.predict(X_v)
            val_loss = mean_squared_error(y_v, val_pred) / 2
            val_loss_history.append(val_loss)

        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_history, label='Training Loss')
        plt.plot(val_loss_history, label='Validation Loss')
        plt.title('MLP Training vs Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (0.5 * MSE)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figures_dir, 'regression_nn_loss.png'))
        plt.close()

        final_params = grid_search.best_params_.copy()
        final_params.pop("max_iter", None)
        final_model = MLPRegressor(
            **final_params,
            max_iter=100,
            early_stopping=False,
            random_state=random_seed,
            verbose=False
        )
        final_model.fit(X_train_preprocessed, y_train)
        y_train_pred = final_model.predict(X_train_preprocessed)
        y_pred = final_model.predict(X_test_preprocessed)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("train_r2", train_r2)

        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_r2", test_r2)

        mlflow.sklearn.log_model(final_model, artifact_path="final_model")

        mlflow.set_tag("model_type", "MLPRegressor_regression_nn")

        residuals = y_test.values - y_pred
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
        plt.axvline(0, color='red', linestyle='--', linewidth=1)
        plt.xlabel('Residual (y_test - y_pred)')
        plt.ylabel('Frequency')
        plt.title('Residual Histogram for MLP Regressor on Test Set')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(figures_dir, 'regression_nn_residuals.png'))
        plt.close()

        metrics = {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2
        }

        return metrics

