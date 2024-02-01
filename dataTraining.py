import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
data = pd.read_csv('solubility_data.csv')
x = data[['Molecular Weight', 'Polar Surface Area', 'HBD', 'HBA', '#Rotatable Bonds', 'Aromatic Rings', 'Heavy Atoms']]
y = data['AlogP']

N = len(x)
print(N)

# Define a numpy array containing ones and concatenate it with the feature matrix
ones = np.ones(N)
Xp = np.c_[ones, x]

# Reshape the target variable (y) to be a column vector
y = y.values.reshape(-1, 1)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(Xp, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize the weights (coefficients) with random values
w = 2 * np.random.rand(8, 1) - 1  # Shape (8, 1) for 7 features + bias term


# Function to train the model and return the trained model
def train_model(X_train, y_train):
    # Initialize the weights (coefficients) with random values for training data
    w = 2 * np.random.rand(8, 1) - 1  # Shape (8, 1) for 7 features + bias term

    # Define the number of training epochs and the learning rate
    epochs = 100000
    learning_rate = 0.00001

    # Training loop
    for epoch in range(epochs):
        # Calculate the predicted values using the current weights for training data
        y_train_predicted = X_train @ w

        # Calculate the error for training data
        train_error = y_train - y_train_predicted

        # Calculate the gradient of the loss with respect to the weights for training data
        train_gradient = -(1 / X_train.shape[0]) * X_train.T @ train_error

        # Update the weights using gradient descent for training data
        w = w - learning_rate * train_gradient

        # Calculate the mean squared error (L2 loss) for training data
        train_L2 = 0.5 * np.mean(train_error ** 2)

        # Print progress every 10% of the epochs
        if epoch % (epochs / 10) == 0:
            print(f"Epoch {epoch}: Training L2 Loss = {train_L2}")
            # Create a trained Linear Regression model
            reg = LinearRegression()
            reg.coef_ = w[1:].T  # Set the coefficients
            reg.intercept_ = w[0][0]  # Set the intercept

            return reg

        # Train the model using your training data
        trained_model = train_model(X_train, y_train)

        # Save the trained model to use later
        joblib.dump(trained_model, 'trained_model.pkl')