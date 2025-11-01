import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Number of samples (200,000)
n_samples = 200000

# Define categorical options for soil type and crop planted
soil_types = ['Sandy', 'Clay', 'Loamy', 'Silty', 'Peaty', 'Chalky']
crops = ['Wheat', 'Rice', 'Maize', 'Barley', 'Cotton', 'Sugarcane', 'Potato']

# Generate categorical features
soil_type_data = np.random.choice(soil_types, n_samples)
crop_data = np.random.choice(crops, n_samples)

# Generate numerical features
crop_age = np.random.randint(10, 181, n_samples)  # Crop age (days) between 10 and 180
humidity = np.clip(np.random.normal(loc=70, scale=10, size=n_samples), 0, 100)  # Humidity in %
temperature = np.clip(np.random.normal(loc=25, scale=5, size=n_samples), 0, 50)  # Temperature in °C
water_level = np.clip(np.random.normal(loc=50, scale=20, size=n_samples), 0, None)  # Water level in cm
rainfall = np.clip(np.random.normal(loc=100, scale=30, size=n_samples), 0, None)  # Rainfall (mm)

# Define adjustment factors for Soil Type and Crop Planted for water required (ml)
soil_adjustment = {
    'Sandy': 20,
    'Clay': -10,
    'Loamy': 0,
    'Silty': -5,
    'Peaty': -15,
    'Chalky': 10
}

crop_adjustment = {
    'Wheat': 0,
    'Rice': 30,
    'Maize': 10,
    'Barley': -5,
    'Cotton': 15,
    'Sugarcane': 40,
    'Potato': 5
}

# Map the categorical values to their corresponding adjustments
soil_adj = np.array([soil_adjustment[s] for s in soil_type_data])
crop_adj = np.array([crop_adjustment[c] for c in crop_data])

# Generate random noise
noise = np.random.normal(loc=0, scale=5, size=n_samples)

# Compute Water Required (ml) using the synthetic relation
water_required = (
    50 +                              # Base intercept
    0.5 * crop_age +
    soil_adj +
    crop_adj +
    (-0.3 * humidity) +
    (0.7 * temperature) +
    (-0.2 * water_level) +
    (-0.1 * rainfall) +
    noise
)

# Ensure water required is non-negative
water_required = np.clip(water_required, 0, None)

# Create a DataFrame with the features and target
df = pd.DataFrame({
    "Crop Age (days)": crop_age,
    "Soil Type": soil_type_data,
    "Crop Planted": crop_data,
    "Humidity (%)": humidity,
    "Temperature (°C)": temperature,
    "Water Level (cm)": water_level,
    "Rainfall (mm)": rainfall,
    "Water Required (ml)": water_required
})

# Optionally, save the dataset to a CSV file
df.to_csv("synthetic_water_requirement_data.csv", index=False)
print("Dataset generated and saved to 'synthetic_water_requirement_data.csv'")
