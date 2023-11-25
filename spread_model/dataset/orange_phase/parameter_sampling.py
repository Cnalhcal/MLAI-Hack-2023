import pandas as pd
import numpy as np 
from scipy.stats import qmc
import sys
import os

#Retrieve parent folder for the spread_function

parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent_dir)

from spread_function import calculate_fire_spread_rate

# ORANGE PHASE:
# Define the bounds for each parameter in the orange phase
bounds_orange_phase = np.array([[11.5, 34.1],  # Wind speed
                   [0, 6],        # Fuel load
                   [66, 100],      # Moisture content
                   [0, 30]])      # Slope angle

# Using Latin Hypercube Sampling (LHS) for a more structured approach
sampler = qmc.LatinHypercube(d=4)
sample = sampler.random(n=1000)
sample = qmc.scale(sample, bounds_orange_phase[:, 0], bounds_orange_phase[:, 1])

# Extracting the samples for each parameter
wind_speed_samples = sample[:, 0]
fuel_samples = sample[:, 1]
moisture_content_samples = sample[:, 2]
slope_angle_samples = sample[:, 3]

# Calculating spread rates for the sampled data
spread_rates = [calculate_fire_spread_rate(wind_speed, moisture, slope, fuel) 
                for wind_speed, moisture, slope, fuel in zip(wind_speed_samples, moisture_content_samples, slope_angle_samples, fuel_samples)]

# Creating a DataFrame to hold the sampled data and calculated spread rates
data_lhs = pd.DataFrame({
    'Wind Speed (km/hr)': wind_speed_samples,
    'Fuel Load (tonnes/hectare)': fuel_samples,
    'Moisture Content (%)': moisture_content_samples,
    'Slope Angle (degrees)': slope_angle_samples,
    'Spread Rate': spread_rates
})

data_lhs.head()
