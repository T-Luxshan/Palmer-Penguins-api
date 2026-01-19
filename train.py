from palmerpenguins import load_penguins
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
import numpy as np

# Load dataset
penguins = load_penguins()

# Clean: drop rows missing key columns
df = penguins.dropna(subset=[
    'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'species'
]).copy()

# Features (4 numeric columns)
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
X = df[features].values

# Create numeric target (0, 1, 2)
species_unique = sorted(df['species'].unique())           # ['Adelie', 'Chinstrap', 'Gentoo']
species_to_code = {name: idx for idx, name in enumerate(species_unique)}
df['species_code'] = df['species'].map(species_to_code)
y = df['species_code'].values.astype(int)

# Train the model with numeric labels
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model + class names (in the correct order)
with open("penguin_model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "class_names": species_unique
    }, f)

print(f"Model trained on {len(df)} penguins")
print("Class mapping:", {v: k for k, v in species_to_code.items()})
print("Classes saved in order:", species_unique)
print("Model saved as penguin_model.pkl")