import numpy as np
import torch
from torch.utils.data import Dataset


def calculate_fire_spread_rate(wind_speed_at_10m, moisture_content, slope_angle, fuel_load=2):
    """
    Calculate the rate of spread of a forest fire.

    Parameters:
    wind_speed_at_10m (float): Wind speed at 10 meters height in km/hr.
    moisture_content (float): Moisture content in percentage (35% - 65%).
    slope_angle (float): Slope angle in degrees.
    fuel_load (float): Fuel load in tonnes/hectare (< 6 mm diameter), default is 2 tonnes/hectare.

    Returns:
    float: Rate of spread of the fire in unspecified units.
    """
    # Constants
    a = 1.674
    b = 0.1798
    c = 0.22
    d = 0.158
    e = -0.227
    f = 0.0662

    # Convert wind speed at 10m to wind speed at 1.5m
    wind_speed_at_1_5m = a + b * wind_speed_at_10m

    # Calculate the rate of spread on flat ground
    rate_flat_ground = c * fuel_load * (wind_speed_at_1_5m ** d) * (moisture_content ** e)

    # Adjust rate for slope
    rate_on_slope = rate_flat_ground * (slope_angle ** f)

    return rate_on_slope

# Example usage
spread_rate = calculate_fire_spread_rate(20, 50, 10)
spread_rate

#################################################################################################


#################################################################################################

def generate_features():
    features = {
        'fuel_age': np.random.uniform(0, 30),
        'wind_speed': np.random.normal(50, 25),
        'temperature': np.random.normal(20, 10),
        'humidity': np.random.normal(50, 20),
        'fuel_moisture_content': np.random.normal(15, 7),
        'slope': np.random.uniform(0, 45),
        'surface_fuel_hazard': np.random.uniform(1, 10),
        'near_surface_fuel_hazard': np.random.uniform(1, 10),
        'near_surface_fuel_height': np.random.normal(50, 25),
        'elevated_fuel_height': np.random.normal(100, 50),
        'flame_height': np.random.normal(25, 12.5),
    }
    return features

def bushfire_rate_of_spread(features):
    # This is a hypothetical function.
    rate = (
        features['wind_speed'] * 0.3 +
        features['temperature'] * 0.2 -
        features['humidity'] * 0.15 +
        features['fuel_moisture_content'] * 0.1 +
        features['slope'] * 0.05 +
        features['surface_fuel_hazard'] * 0.1 +
        features['near_surface_fuel_hazard'] * 0.1 -
        features['near_surface_fuel_height'] * 0.05 +
        features['elevated_fuel_height'] * 0.05 +
        features['flame_height'] * 0.1
    )
    return np.clip(rate, 0, None)  # Ensuring rate of spread is non-negative

def generate_bushfire_data(size):
    data = []
    for _ in range(size):
        features = generate_features()
        rate_of_spread = bushfire_rate_of_spread(features)
        features['rate_of_spread'] = rate_of_spread
        data.append(features)
    return data

# Example usage
bushfire_data = generate_bushfire_data(100000)
