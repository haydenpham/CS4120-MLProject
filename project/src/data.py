import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from utils import classify_price, get_median_price


def load_and_engineer_features(data_path, random_state=42):
    """
    Load data and perform feature engineering (remove redundant features and group related ones).
    """
    # Load the data
    df = pd.read_csv(data_path)

    # Calculate median price
    median_price = get_median_price(df)
    print(f"\nMedian Sale Price: ${median_price:,.2f}")


    # Combine bathroom features
    df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath'] +
                       df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
    df = df.drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], axis=1)

    # Combine porch features into one
    df['TotalPorchSF'] = (df['OpenPorchSF'] + df['EnclosedPorch'] +
                          df['3SsnPorch'] + df['ScreenPorch'])
    df = df.drop(['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'], axis=1)

    # Drop low-variance features
    df = df.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating'], axis=1)

    # Drop Year/Month sold (temporal features may cause overfitting)
    df = df.drop(['MoSold', 'YrSold'], axis=1)


    df['OverallScore'] = df['OverallQual'] * df['OverallCond']

    # Remove redundant features
    df = df.drop(['GarageType', 'GarageFinish', 'Fence', 'Alley'], axis=1)
    df = df.drop(['LowQualFinSF', 'PoolArea', 'MiscVal'], axis=1)
    df = df.drop(['GarageQual', 'GarageCond'], axis=1)
    df = df.drop(['PoolQC', 'MiscFeature'], axis=1)

    # Target Variable: Above/Below Median Price Classification
    df['PriceCategory'] = df['SalePrice'].apply(lambda x: classify_price(x, median_price))

    print("\nPriceCategory distribution:")
    print(f"Below Median (0): {(df['PriceCategory'] == 0).sum()}")
    print(f"Above Median (1): {(df['PriceCategory'] == 1).sum()}")

    # Remove Id and SalePrice columns
    df = df.drop(['Id', 'SalePrice'], axis=1)

    # Separate features and target
    X = df.drop('PriceCategory', axis=1)
    y = df['PriceCategory']

    return X, y, median_price


def get_preprocessing_pipeline(X):
    """
    Create preprocessing pipeline: fill missing values, encode categorical variables, and remove low-variance features.
    """
    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Pre-processing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('variance_filter', VarianceThreshold(threshold=0.01))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor, numeric_features, categorical_features


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)