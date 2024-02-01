import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
# Load the dataset using pandas with the specified delimiter
df = pd.read_csv("All Drugs.csv", delimiter=';', on_bad_lines='skip')

# Check if the file is loaded successfully
if not df.empty:
    print(f"Dataset loaded successfully.")
else:
    print(f"Failed to load the dataset.")

# Check the column names in the DataFrame
print(df.columns)
# Filter the DataFrame to include only rows where Type is "Small molecule"
small_molecules_df = df[df['Type'] == 'Small molecule']

# Check if any rows match the filter condition
if not small_molecules_df.empty:
    print(f"Filtered DataFrame contains {len(small_molecules_df)} rows with Type 'Small molecule'.")
else:
    print("No rows match the filter condition.")

# Save the filtered DataFrame as a new CSV file called "small_molecules.csv"
small_molecules_df.to_csv("/kaggle/working/small_molecules.csv", index=False)

print("Filtered DataFrame saved as 'small_molecules.csv'.")


# Read the small_molecules.csv file
small_molecules_df = pd.read_csv("small_molecules.csv")

# List of columns to keep
columns_to_keep = [
    "Molecular Weight",
    "AlogP",
    "Polar Surface Area",
    "HBD",
    "HBA",
    "#Rotatable Bonds",
    "Aromatic Rings",
    "Heavy Atoms",
]


# Filter the DataFrame to include only the selected columns
filtered_small_molecules_df = small_molecules_df[columns_to_keep]

# Check if any rows match the filter condition
if not filtered_small_molecules_df.empty:
    print(f"Filtered DataFrame contains {len(filtered_small_molecules_df)} rows with selected columns.")
else:
    print("No rows match the filter condition.")

# Save the filtered DataFrame as a new CSV file called "solubility_data.csv"
filtered_small_molecules_df.to_csv("solubility_data.csv", index=False)

print("Filtered DataFrame with selected columns saved as 'solubility_data.csv'.")

#Display the list of columns and the data
data = pd.read_csv('solubility_data.csv')
x = data[['Molecular Weight', 'Polar Surface Area', 'HBD', 'HBA', '#Rotatable Bonds', 'Aromatic Rings', 'Heavy Atoms']]
# Check the shape of the feature matrix (X)
print(x.shape)


# Convert the specified columns to float64
columns_to_convert = ['AlogP', 'Polar Surface Area', 'HBD', 'HBA', '#Rotatable Bonds', 'Aromatic Rings', 'Heavy Atoms']

for column in columns_to_convert:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Calculate missing values
data.isnull().sum()

# To remove rows with missing values (NaN)
data.dropna(inplace=True)

# To impute missing values with the mean
data.fillna(data.mean(), inplace=True)

# Double check that no more missing values
data.isnull().sum()

