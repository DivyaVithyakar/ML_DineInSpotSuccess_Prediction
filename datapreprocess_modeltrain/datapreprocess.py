import pandas as pd
import numpy as np
from functions import quanQual, univaiate, replace_outlier
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv("../data/zomato.csv", encoding='latin1')
print(dataset.isnull().sum())

# Filter for India and drop unused cols
dataset = dataset[dataset['Country Code'] == 1]
columns_to_drop = [
    'Restaurant ID', 'Restaurant Name', 'Address', 'Locality Verbose',
    'Rating color', 'Rating text', 'Is delivering now', 'Switch to order menu', 'Country Code'
]
dataset.drop(columns=columns_to_drop, inplace=True)

# Remove rows missing ratings, create target
dataset = dataset[~dataset['Aggregate rating'].isnull()]
dataset['Successful'] = dataset['Aggregate rating'].apply(lambda x: 1 if x > 4.0 else 0)

# Encode binary categorical features
dataset['Has Table booking'] = dataset['Has Table booking'].apply(lambda x: 1 if x == 'Yes' else 0)
dataset['Has Online delivery'] = dataset['Has Online delivery'].apply(lambda x: 1 if x == 'Yes' else 0)

# Identify quantitative/qualitative features
quan, qual = quanQual(dataset)
print(quan, qual)

# Univariate stats
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
uni_describe = univaiate(dataset, quan)
print(uni_describe)

# Replace outliers
df = replace_outlier(dataset, quan, uni_describe)

# Log transform for high skew
log_transform_cols = [
    col for col in uni_describe
    if "Skew" in uni_describe[col]
       and uni_describe[col]["Skew"] > 1
       and (df[col] >= 0).all()
]
post_describe = univaiate(df, quan)
print("Skew after transform:")
for col in log_transform_cols:
    print(f"{col}: {post_describe[col]['Skew']:.2f}")

    # Save the cleaned dataset to a new CSV file
df.to_csv("../data/zomato-cleaned.csv", index=False)

# Compute correlation matrix (only for numeric columns)
numeric_columns = df.select_dtypes(include=[np.number]).columns  # Select numeric columns
correlation_matrix = df[numeric_columns].corr()

# Set the figure size for the heatmap
plt.figure(figsize=(12, 8))

# Create the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Add title and display the heatmap
plt.title("📊 Feature Correlation Heatmap - Zomato Dataset")
plt.tight_layout()
plt.show()