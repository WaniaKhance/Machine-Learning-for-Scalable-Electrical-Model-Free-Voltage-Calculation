# Machine-Learning-for-Scalable-Electrical-Model-Free-Voltage-Prediction ⚡


## Project Description

This project focuses on analyzing power grid data and predicting voltage magnitudes using various machine learning techniques. The main objectives of the project are:

1. Generate and aggregate power consumption data for different meters and buses.
2. Calculate voltage magnitudes for each bus.
3. Perform time-based aggregation of the data.
4. Implement and compare different machine learning models for voltage prediction.

## Data Privacy

This project involves proprietary data from a corporate partner. Out of respect for the organization's confidentiality and intellectual property rights, the specific dataset used in this analysis cannot be publicly shared

## Features

- Data generation with high demand simulation
- PQ (Power and Reactive Power) data aggregation
- Voltage data calculation and aggregation
- Time-based data aggregation
- Machine learning models for voltage prediction:
  - Random Forest Regressor
  - XGBoost Regressor
  - Artificial Neural Network (ANN)

## Dependencies

- pandas
- sqlite3
- numpy
- random
- matplotlib
- scikit-learn
- xgboost
- tensorflow (for ANN model)
- keras (for ANN model)

## Usage

1. Ensure all dependencies are installed.
2. Place the `net.db` SQLite database file in the project directory.
3. Run the main script: python Project_Final_code.py


## Data Processing Steps

1. Generate high-demand data for meter profiles
2. Aggregate PQ data for each bus and phase
3. Calculate and aggregate voltage data
4. Combine PQ and voltage data for each bus
5. Perform time-based aggregation across all buses

## Machine Learning Models

The project implements three different machine-learning models for voltage prediction:

1. Random Forest Regressor
2. XGBoost Regressor
3. Artificial Neural Network (ANN)

The models use power consumption data (P and Q) as input features to predict voltage magnitudes for three phases.

## Results

The script outputs the performance metrics for each model:

- Root Mean Square Error (RMSE) for Random Forest and XGBoost
- Mean Absolute Error (MAE) for ANN

## Future Improvements

- Implement hyperparameter tuning for the machine learning models
- Add more advanced feature engineering techniques
- Explore other machine learning algorithms for comparison
- Develop a user interface for easier data visualization and model interaction

