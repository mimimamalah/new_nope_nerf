from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np

# Load depth maps
with np.load("data/CVLab/hubble_output_resized/dpt/depth_0000.npz") as npz_file1:
    depths1 = npz_file1['pred']

with np.load("data/CVLab/hubble_output_resized_paper/dpt/depth_0000.npz") as npz_file2:
    depths2 = npz_file2['pred']

X = depths2.flatten().reshape(-1, 1)  # Input
y = depths1.flatten()  # Target

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a polynomial regression model
degree = 3  # Degree of the polynomial
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Train the model
model.fit(X_train, y_train)

# Validate the model (optional, but recommended)
print(f"Validation Score: {model.score(X_val, y_val)}")

# Use the model to convert all distances from the first dataset
converted_distances = model.predict(X).reshape(depths1.shape)

# Save the converted distances
np.savez("converted_distances_complex.npz", pred=converted_distances)
