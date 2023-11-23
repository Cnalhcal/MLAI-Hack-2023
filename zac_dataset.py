import math
import numpy as np
# import torch
# from torch.utils.data import Dataset

def generate_features():
    """
    Features data for factors affecting bushfires
    (Refer to details according to
    https://www.ga.gov.au/education/natural-hazards/bushfire)
    :return: dict
    """
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

#mathematical implementations of params
def fdi(degCure, AirTemp, RelHumid, AvgWind):
    """
    Mk3 Model (Noble, 1980)
    Grassland Fire Danger Index
    https://www.bushfirecrc.com/sites/default/files/managed/resource/fire_knowledge_synthesis_final_report.pdf
    :param degCure: degree of curing (%)
    :param AirTemp: air temperature (degrees Celsius)
    :param RelHumid: relative humidity (%)
    :param AvgWind: wind speed (km/h) as measured/estimated at a height of 10m in the open
    :return:
    """
    #INSERT INPUT VALIDATION HERE
    return 2 * math.exp(-23.6 + 5.01 * math.log(degCure)
                        + 0.0281 * AirTemp
                        - 0.226 * math.sqrt(RelHumid)
                        + 0.633 * math.sqrt(AvgWind))

#Rate of Fire Spread (kmh^-1)
def RoFS(FDI):
    """
    Headfire rate of spread in km/h
    :param FDI:
    :return:
    """
    return 0.13 * FDI

def rate_of_spread_level_ground(wind_speed,
                                fuel_moisture_content):
    """
    Rate of spread on level ground (R, m/min) burning a standardized fuel
    load (w) of 25 t/ha (equation 2.38)
    :return:
    """
    return 5.492 * math.exp(0.158 + wind_speed - 0.277 * fuel_moisture_content)


def rate_of_spread_adjusted_fuel_load(fuel_load,
                                      wind_speed,
                                      fuel_moisture_content):
    return (0.04 * fuel_load
            * rate_of_spread_level_ground(wind_speed,
                                          fuel_moisture_content))

def rate_of_spread_predict(fuel_load, wind_speed, fuel_moisture_content):
    return (0.22 * fuel_load
            * math.exp(0.158 * wind_speed - 0.227 * fuel_moisture_content))


def rate_of_spread_sloping_ground(fuel_load,
                                  wind_speed,
                                  fuel_moisture_content,
                                  theta):
    """
    [equation 2.41]
    :param fuel_load:
    :param wind_speed:
    :param fuel_moisture_content:
    :param theta: slope angle in degrees
    :return:
    """
    return (rate_of_spread_predict(fuel_load,
                                  wind_speed,
                                  fuel_moisture_content)
            * math.exp(0.0662 * theta))


if __name__ == "__main__":
    # Example usage
    bushfire_data = generate_bushfire_data(100000)


