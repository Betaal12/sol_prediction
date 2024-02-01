import warnings

import numpy as np
import joblib
model = joblib.load("trained_model (1).pkl")
def predict_A_log_P(model):
    print("Enter the feature values for the molecule:")
    try:
        Molecular_Weight = float(input("Molecular Weight: "))
        Polar_Surface_Area = float(input("Polar Surface Area: "))
        HBD = int(input("Number of H-Bond Donors: "))
        HBA = int(input("Number of H-Bond Acceptors: "))
        Rotatable_Bonds = int(input("Number of Rotatable Bonds: "))
        Aromatic_Rings = int(input("Number of Aromatic Rings: "))
        Heavy_Atoms = float(input("Number of Heavy Atoms: "))
    except ValueError:
        print("Invalid input. Please enter valid numeric values for all features.")
        return

    # Create a feature vector from user inputs
    feature_vector = np.array([[Molecular_Weight, Polar_Surface_Area, HBD, HBA, Rotatable_Bonds, Aromatic_Rings, Heavy_Atoms]])

    # Predict A log P using the provided model
    predicted_A_log_P = model.predict(feature_vector)

    print(f"Predicted A log P: {predicted_A_log_P[0][0]:.2f} ")

# Call the predict_A_log_P function with your trained model
try:
    predict_A_log_P(model)
except warnings.warn():
    print("Use version 1.2.2 of scikt-learn for accurate results")