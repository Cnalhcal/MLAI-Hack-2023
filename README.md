<h1 style="text-align: centre">Bushfire Prediction Model: Revolutionizing Fire Management in Australia</h1>

<img src="https://github.com/Cnalhcal/MLAI-Hack-2023/assets/100665526/acd7ef48-64ba-4131-b315-44e40574377d.png" width="500px">

### Overview
Australia's landscapes have always been prone to devastating bushfires, with recent years witnessing some of the most catastrophic events in history. Our project, the Bushfire Prediction Model, aims to make a significant impact in this challenging context. Developed for a hackathon, this model leverages the power of deep learning to predict the spread of bushfires, providing crucial information that can be used for effective fire management and emergency response planning.

Click this image to check out our video walkthrough of the project:

[![Link to the video explanation of the project](https://img.youtube.com/vi/fPk_l4gE4S0/0.jpg)](https://www.youtube.com/watch?v=fPk_l4gE4S0)


### Impact in Australia
Bushfires in Australia pose a significant threat to both natural ecosystems and human settlements. Our model's ability to accurately predict the spread of these fires is a game-changer. It empowers firefighters, emergency response teams, and policy makers with the insights needed to allocate resources effectively, plan evacuation routes, and take preemptive actions to mitigate the impact of these potentially deadly natural disasters.

### Innovation in Deep Learning
Our approach to designing a neural network from scratch showcases a deep understanding of the fundamentals of deep learning. By building the model from the ground up, we have been able to tailor it specifically to the unique challenges and nuances of bushfire prediction. This custom approach stands in contrast to the use of off-the-shelf models and demonstrates a thorough grasp of machine learning principles and techniques.

### Synthetic Data Generation and Sampling Techniques
Recognizing the limitations in available real-world data on bushfires, we took the innovative step of generating our own synthetic data. This data was meticulously crafted based on extensive research and understanding of bushfire dynamics. We employed sophisticated sampling techniques, ensuring that our dataset robustly represents various scenarios and conditions under which bushfires occur and spread. This thoughtful approach to data generation underpins the reliability and accuracy of our model.

### Technical Implementation
The heart of our project is the BushFireModel [(A demonstration notebook is available on Colab)](https://colab.research.google.com/drive/1ZrslTfcZE-qQW-1vlBR6kUyr5ATm4EUA?usp=sharing), a neural network built using PyTorch. It's structured to process key environmental factors – such as wind speed, moisture content, fuel load, and slope angle – to predict the rate at which bushfires spread. We trained our model using a custom dataset, employing techniques like Latin Hypercube Sampling for data generation and advanced neural network architectures for modeling.

```
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
BushfireModel                            --
├─ModuleList: 1-1                        --
│    └─Linear: 2-1                       1,000
│    └─Linear: 2-2                       20,100
│    └─Linear: 2-3                       101
=================================================================
Total params: 21,201
Trainable params: 21,201
Non-trainable params: 0
=================================================================
```


### User Interface and Visualization
While our project currently does not feature a front-end interface, the potential for integration into user-friendly applications is immense. The model's predictions could be visualized on interactive maps, providing intuitive, real-time insights into bushfire behavior. This aspect opens up avenues for future development and collaboration.

### Conclusion
Our Bushfire Prediction Model stands as a testament to the innovative application of deep learning in addressing real-world problems. It's not just a technical achievement but a tool with the potential to save lives, protect property, and preserve Australia's natural heritage. As we continue to refine and enhance our model, we look forward to contributing to the global effort in combating the challenge of bushfires.

# Spread Function:

This Python function, `calculate_fire_spread_rate`, is designed to estimate the rate of spread of a forest fire under given environmental conditions. The function takes into account several key factors that influence fire behavior in forested areas.

### Parameters

The function accepts the following parameters:

- `wind_speed_at_10m (float)`: The wind speed measured at a height of 10 meters, expressed in kilometers per hour (km/hr).
- `moisture_content (float)`: The moisture content of the environment, given as a percentage. This should be in the range of 35% to 65%.
- `slope_angle (float)`: The angle of the ground slope, expressed in degrees. This parameter accounts for the impact of terrain on fire spread.
- `fuel_load (float, optional)`: The load of combustible material, specifically referring to materials with a diameter less than 6 mm, measured in tonnes per hectare. The default value is 2 tonnes/hectare, which is a typical value for forested areas.

### Constants

The function uses several constants in its calculations:

- `a (1.674)`: A constant used for converting wind speed at 10m to wind speed at 1.5m height.
- `b (0.1798)`: Another constant used for the wind speed conversion.
- `c (0.22)`: A scaling factor for the fuel load in the rate of spread calculation.
- `d (0.158)`: A constant that modulates the influence of wind speed at 1.5m on the rate of spread.
- `e (-0.227)`: A constant that modulates the influence of moisture content on the rate of spread.
- `f (0.0662)`: A constant that modulates the influence of ground slope on the rate of spread.

### Function Mechanics

1. **Wind Speed Conversion**: The function first converts the wind speed measured at 10 meters to an estimated wind speed at 1.5 meters using the formula:<br> `wind_speed_at_1_5m = a + b * wind_speed_at_10m`.

2. **Rate of Spread on Flat Ground**: It then calculates the rate of spread on flat ground using the formula:<br> `rate_flat_ground = c * fuel_load * (wind_speed_at_1_5m ** d) * (moisture_content ** e)`.

3. **Slope Adjustment**: Finally, the function adjusts this rate based on the slope of the ground:<br> `rate_on_slope = rate_flat_ground * (slope_angle ** f)`.

### Usage

This function is useful for forest fire management and prediction purposes. By inputting various environmental conditions, one can estimate the rate at which a forest fire might spread, aiding in effective planning and response strategies.

# Sampling Parameters:

### Sampling Method:
- The script uses Latin Hypercube Sampling (LHS) from the SciPy library. LHS is an efficient statistical method for generating a distributed sample of parameter values from a multidimensional distribution. This ensures a more uniform and representative sample across the parameter space.

### Parameters:
- **Wind Speed:** Ranging from 11.5 to 34.1 km/hr.
- **Fuel Load:** Ranging from 0 to 6 tonnes/hectare.
- **Slope Angle:** Ranging from 0 to 30 degrees.
- **Moisture Content:** Three seperate models were made to account for the three different generalized bands used by the CFA's Grassland Curing Guide.

  <img src="readme_img/SeasonalProgressionofcuringinSEAUS.png">
  
### Data Generation:
- The script generates 40000 samples for each parameter within their defined bounds for each range above (only YELLOW PHASE in collab.
- It calculates the spread rate of forest fires for each set of parameters using a predefined function `calculate_fire_spread_rate`.

### Output:
- The script creates a pandas DataFrame named `data_lhs`, containing the sampled data and the corresponding calculated fire spread rates.
- The DataFrame includes columns for each parameter and the calculated spread rate.

### Usage
To run this script, ensure that all required libraries are installed. The function `calculate_fire_spread_rate` should be defined in your environment or included in the script. This function is essential for calculating the fire spread rate based on the sampled parameters.

### Example Output
The command `data_lhs.head()` at the end of the script displays the first five rows of the generated dataset, providing a glimpse into the structure of the data, which includes the sampled parameters and the calculated spread rates.

### Applications
This dataset can be used in various applications, including:
- Training machine learning models for predicting forest fire spread rates.
- Conducting statistical analysis to understand the impact of environmental factors on forest fire dynamics.
- Assisting in forest fire management and risk assessment studies.

### Customization
Users can adjust the parameter bounds and the sample size according to their specific needs or to explore different environmental scenarios. 

# PyTorch Input Tensor

### Overview
In addition to generating the dataset, this script includes functionality to convert the pandas DataFrame into a PyTorch Dataset. This allows for easy integration with PyTorch's DataLoader for efficient batching, shuffling, and loading during the training of machine learning models.

### Implementation Details

#### FireSpreadDataset Class
- This class is a subclass of `torch.utils.data.Dataset`.
- It takes a pandas DataFrame as input and converts it into a PyTorch tensor.
- The `__len__` method returns the size of the dataset.
- The `__getitem__` method returns a single sample from the dataset, including both features and the target variable. The target variable is assumed to be the last column in the DataFrame.

### Usage
- Instantiate the `FireSpreadDataset` class with the `data_lhs` DataFrame.
- Access individual samples using indexing, which returns both features and target for each sample.

# Bushfire Model

### Overview
The `BushfireModel` class is a PyTorch neural network model designed to predict the spread rate of bushfires based on input features. The model is a fully connected feedforward neural network with customizable hidden layers.

### Features
- **Customizable Architecture**: The number of hidden layers and the number of neurons in each layer can be specified when creating an instance of the model.
- **Activation Function**: Uses the ReLU activation function for non-linear transformations between layers, except for the output layer, which is linear.
- **Flexible Input**: Designed to work with any number of input features.

### Requirements
- Python 3.x
- PyTorch

### Usage
To use the `BushfireModel`, first initialize it with the desired architecture:

```python
import torch
from bushfire_model import BushfireModel

# Example: Create a model with 4 input features and two hidden layers with 500 and 100 neurons respectively
model = BushfireModel(n=4, hidden_layers=[500, 100])
```

# Training the BushfireModel

### Overview
This guide outlines the process for training the `BushfireModel`, a neural network designed to predict the spread rate of bushfires. The model is trained using PyTorch, a popular deep learning library.

### Requirements
- Python 3.x
- PyTorch
- A dataset encapsulated in a PyTorch `Dataset` object, referred to as `fire_spread_data`.

### Training Process
The training process involves the following steps:

1. **Splitting the Dataset**: The dataset is split into training and validation sets, with 80% of the data used for training and the remaining 20% for validation.

2. **Creating DataLoaders**: PyTorch `DataLoader` objects are created for both the training and validation sets to enable batch processing and shuffling of the data.

3. **Model Initialization**: The `BushfireModel` is initialized with a specified number of input features and hidden layers.

4. **Defining Loss Function and Optimizer**: The Mean Squared Error (MSE) loss function and Adam optimizer are used.

5. **Training Loop**: The model is trained over multiple epochs, where each epoch involves a pass over the entire training dataset and a subsequent evaluation on the validation dataset.

```python
# Initialize the model
model = BushfireModel(n=4, hidden_layers=[200,100])

# Define the loss criterion and optimizer
loss_criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Train the model
num_epochs = 50
train_model(model, train_loader, val_loader, loss_criterion, optimizer, num_epochs)
```
# Prediction Using the Trained BushfireModel

## Overview
This guide provides instructions on how to use the trained `BushfireModel` to make predictions. The `predict` function is designed to take a model and a set of input features and return the model's predictions.

## Function Description

### `predict` Function
- **Purpose**: To make predictions using the trained `BushfireModel`.
- **Parameters**:
  - `model`: The trained PyTorch model.
  - `input_features`: A tensor of input features on which predictions are to be made.
- **Returns**: The predicted values as output by the model. We theh print them with the actual values and see how well model works.

### Code Snippet
```python
def predict(model, input_features):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = model(input_features)
    return predictions
```
