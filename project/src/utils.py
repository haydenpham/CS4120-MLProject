import pandas as pd

file_path = "../data/train.csv"

# For the classification task, we want to divide the SalePrice column in our dataset
# into 3 groups: low (bottom 33%), medium (mid 33%), and high (top 33%)

# We now need to find the 33rd and 67th percentiles of the data
def load_and_calculate_percentiles():
    """
    Load the training data and calculate the 33rd and 67th percentiles of SalePrice
    to divide it into 3 equal groups for classification.
    """
    train_df = pd.read_csv(file_path)

    # Calculate the 33rd and 67th percentiles of SalePrice
    percentile_33 = train_df['SalePrice'].quantile(0.33)
    percentile_67 = train_df['SalePrice'].quantile(0.67)

    return int(percentile_33), int(percentile_67)

def classify_price_into_3(price):
    """
    Classify the given price into 'low', 'medium', or 'high' based on the provided percentiles.
    """
    # found in load_and_calculate_percentiles()
    low = 139000
    high = 191000
    if price < low:
        return '0'
    elif price > high:
        return '2'
    else:
        return '1'

def classify_price_text(price):
    """
    Classify the given price into 'low', 'medium', or 'high' based on the provided percentiles.
    """
    # found in load_and_calculate_percentiles()
    low = 139000
    high = 191000
    if price <= low:
        return 'Low'
    elif price >= high:
        return 'High'
    else:
        return 'Medium'


def classify_price(price, median_price=None):
    """
    Binary classification: above-median price classification.

    Parameters:
    -----------
    price : float
        The sale price of the house
    median_price : float, optional
        Median price for classification. If None, must be set externally.

    Returns:
    --------
    int
        0 for below median, 1 for above median
    """
    if median_price is None:
        raise ValueError("median_price must be provided for classification")

    return 1 if price >= median_price else 0


def get_median_price(df, price_column='SalePrice'):
    return df[price_column].median()