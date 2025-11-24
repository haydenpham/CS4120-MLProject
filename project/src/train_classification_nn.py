import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from utils import get_median_price, classify_price
import matplotlib.pyplot as plt
import mlflow.sklearn
import warnings
import os
warnings.filterwarnings('ignore')

random_seed = 42


def load_and_engineer_classification_features(data_path):
    df = pd.read_csv('../data/train.csv')

    # Step 2: Process Data
    median_price = get_median_price(df)
    df['PriceCategory'] = df['SalePrice'].apply(lambda x: classify_price(x, median_price))

    # Remove Id and SalePrice columns
    df = df.drop(['Id', 'SalePrice'], axis=1)
    # Combine bathroom features
    df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    df = df.drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], axis=1)

    # Combine porch features into one
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    df = df.drop(['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'], axis=1)

    # Drop low-variance features
    df = df.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating'], axis=1)

    # Drop Year/Month sold (temporal features may cause overfitting)
    df = df.drop(['MoSold', 'YrSold'], axis=1)

    # Combine quality features
    df['OverallScore'] = df['OverallQual'] * df['OverallCond']
    # Redundant features
    df = df.drop(['GarageType', 'GarageFinish', 'Fence', 'Alley'], axis=1)
    df = df.drop(['LowQualFinSF', 'PoolArea', 'MiscVal'], axis=1)
    df = df.drop(['GarageQual', 'GarageCond'], axis=1)
    df = df.drop(['PoolQC', 'MiscFeature'], axis=1)

    # Separate features and target
    X = df.drop('PriceCategory', axis=1)
    y = df['PriceCategory']


    return X, y


def train_classification_nn(random_seed=42):
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
    mlflow_tracking_uri = os.path.join(os.path.dirname(__file__), '..', 'mlruns')
    figures_dir = os.path.join(os.path.dirname(__file__), '..', 'figures_final')

    os.makedirs(figures_dir, exist_ok=True)

    # Change to \\ if on Windows
    if os.name == 'nt':
        mlflow.set_tracking_uri(f"file:\\{mlflow_tracking_uri}")
    else:
        mlflow.set_tracking_uri(f"file://{mlflow_tracking_uri}")

    mlflow.end_run()
    mlflow.set_experiment("classification_nn")

    X, y = load_and_engineer_classification_features(data_path)

    # Split the data: 75% training, 25% testing
    X_main, X_test, y_main, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_main, y_main, test_size=0.25, random_state=random_seed
    )

    X_train['MSSubClass'] = X_train['MSSubClass'].astype(str)
    X_val['MSSubClass'] = X_val['MSSubClass'].astype(str)
    X_test['MSSubClass'] = X_test['MSSubClass'].astype(str)

    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Numerical pipeline
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine both pipelines
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    # Fit preprocessor once on training data to avoid data leakage
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_val_preprocessed = preprocessor.transform(X_val)
    X_test_preprocessed = preprocessor.transform(X_test)


    base_mlp = MLPClassifier(
        max_iter=500,
        random_state=random_seed,
        early_stopping=True
    )
    # Best params found: {'activation': 'relu', 'alpha': 1.0, 'hidden_layer_sizes': (32, 16), 'learning_rate_init': 0.0005, 'solver': 'adam'}
    param_grid = {
        "hidden_layer_sizes": [(32,),
                               (32, 16),
                               (16, 8),
                               (64, 8), ],
        "activation": ["relu", "logistic"],
        "alpha": [0.1, 1.0, 5.0],
        "solver": ["adam"],
        "learning_rate_init": [0.0001, 0.0005],
    }

    # Grid search with 4-fold CV
    grid_search = GridSearchCV(
        estimator=base_mlp,
        param_grid=param_grid,
        cv=4,
        scoring="accuracy",
    )

    grid_search.fit(X_train_preprocessed, y_train)
    best_params = grid_search.best_params_


    with mlflow.start_run(run_name="classification_nn_final_model"):
        mlflow.log_params(best_params)
        mlflow.log_param("final_max_iter", 150)
        mlflow.log_param("random_seed", random_seed)
        mlp_model_to_use = MLPClassifier(
            **best_params,
            max_iter=1,
            random_state=random_seed,
            warm_start=True,
            early_stopping=False,
            verbose=False,
            batch_size=64
        )

        train_accuracy_history = []
        val_accuracy_history = []

        n_epochs = 150
        mlflow.log_param("n_epochs", n_epochs)

        for epoch in range(n_epochs):
            mlp_model_to_use.fit(X_train_preprocessed, y_train)

            # Training Accuracy
            y_train_pred_mon = mlp_model_to_use.predict(X_train_preprocessed)
            train_accuracy_mon = accuracy_score(y_train, y_train_pred_mon)
            train_accuracy_history.append(train_accuracy_mon)

            y_val_pred_mon = mlp_model_to_use.predict(X_val_preprocessed)
            val_accuracy_mon = accuracy_score(y_val, y_val_pred_mon)
            val_accuracy_history.append(val_accuracy_mon)

        plt.figure(figsize=(10, 6))
        plt.plot(train_accuracy_history, label='Training Accuracy', color='blue')
        plt.plot(val_accuracy_history, label='Validation Accuracy', color='orange', linestyle='--')

        plt.title('Learning Curve: Classification NN (Training vs Validation)', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        mlflow.log_figure(plt.gcf(), 'plot1.png')
        plt.savefig(os.path.join(figures_dir, 'classification_nn_loss.png'))
        plt.close()


        # Train the final model with optimal number of epochs
        optimal_epochs = 40

        final_model = MLPClassifier(
            **best_params,
            max_iter=optimal_epochs,
            random_state=random_seed,
            early_stopping=False,
            verbose=False,
        )

        final_model.fit(X_train_preprocessed, y_train)

        mlflow.log_param("final_epochs", optimal_epochs)

        # Use the trained model for predictions
        y_train_pred = final_model.predict(X_train_preprocessed)
        y_test_pred = final_model.predict(X_test_preprocessed)

        # Calculate metrics for training set
        train_accuracy = accuracy_score(y_train, y_train_pred)

        train_f1 = f1_score(y_train, y_train_pred, average='micro')

        # Calculate metrics for test set
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='micro')

        # Log metrics to MLFlow
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1", test_f1)

        # Log the model
        mlflow.sklearn.log_model(final_model, "model")
        mlflow.end_run()

        metrics = {
            'train_accuracy': train_accuracy,
            'train_f1': train_f1,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
        }

        return metrics

