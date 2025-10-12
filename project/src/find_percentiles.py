# For the classification task, we want to divide the SalePrice column in our dataset
# into 3 groups: low (bottom 33%), medium (mid 33%), and high (top 33%)

# We now need to find the 33rd and 67th percentiles of the data

# Prompt: Write code to read from "../data/train.csv" and
#   find the 33rd and 67th percentiles of the "salesPrice" column (last column) to divide it into 3 equal parts.
import pandas as pd

def load_and_calculate_percentiles():
    """
    Load the training data and calculate the 33rd and 67th percentiles of SalePrice
    to divide it into 3 equal groups for classification.

    Returns:
        tuple: (33rd_percentile, 67th_percentile, dataframe)
    """
    # Read the training data
    train_df = pd.read_csv("../data/train.csv")

    # Calculate the 33rd and 67th percentiles of SalePrice
    percentile_33 = train_df['SalePrice'].quantile(0.33)
    percentile_67 = train_df['SalePrice'].quantile(0.67)

    print(f"Dataset shape: {train_df.shape}")
    print(f"SalePrice statistics:")
    print(f"  Min: ${train_df['SalePrice'].min():,.0f}")
    print(f"  33rd percentile: ${percentile_33:,.0f}")
    print(f"  Median (50th): ${train_df['SalePrice'].median():,.0f}")
    print(f"  67th percentile: ${percentile_67:,.0f}")
    print(f"  Max: ${train_df['SalePrice'].max():,.0f}")

    # Create classification labels based on percentiles
    def classify_price(price):
        if price <= percentile_33:
            return 'low'
        elif price <= percentile_67:
            return 'medium'
        else:
            return 'high'

    train_df['PriceCategory'] = train_df['SalePrice'].apply(classify_price)

    # Show distribution of categories
    category_counts = train_df['PriceCategory'].value_counts().sort_index()
    print(f"\nPrice category distribution:")
    for category, count in category_counts.items():
        percentage = (count / len(train_df)) * 100
        print(f"  {category}: {count} houses ({percentage:.1f}%)")

    return percentile_33, percentile_67, train_df

if __name__ == "__main__":
    p33, p67, data = load_and_calculate_percentiles()
