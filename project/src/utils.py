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

def classify_price(price):
    """
    Classify the given price into 'low', 'medium', or 'high' based on the provided percentiles.
    """
    # found in load_and_calculate_percentiles()
    low = 139000
    high = 191000
    if price <= low:
        return '0'
    elif price >= high:
        return '2'
    else:
        return '1'