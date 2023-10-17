import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data(path):
    """Load the csv file which contains the dataset

    Args:
        path (str): directory of the dataset

    Returns:
        DataFrame: the dataset loaded into a pandas DataFrame object.
        The dataset contains the following columns:
        - volume: the volume of the car
        - curbweight: the weight of the car without additional load
        - peakrpm: the peak speed of the car
        - citympg: the city milage of the car
        - horsepower: the horse power of the car in 
        - fueltpye_diesel: True if fueltype is diesel
        - fueltype_gas: True if fueltype is gas
        - price: the price of the car
    """
    df = pd.read_csv(path) 
    return df

def  split_data(df):
    """split the dataset into train and test data

    Args:
        df (DataFrame): the dataset loaded into a pandas DataFrame object.

    Returns:
        DataFrame: X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test are the training and testing data sets.
        X_train and X_test are the input features and y_train and y_test are the target values.
        The input features are the columns volume, curbweight, peakrpm, citympg, horsepower, fueltype_diesel, fueltype_gas.
        The target values are the column price.
    """
    X = df.iloc[:, :-1]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalise data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
    """training a random forest regression model

    Args:
        X_train (DataFrame): features of the train data
        X_test (DataFrame): features of the test data
        y_train (DataFrame): target of the train dataset
        y_test (DataFrame): target of the train dataset

    Returns:
        model: the trained regression model
    """
    # Initialize and train a Random Forest Regressor
    random_forest_model = RandomForestRegressor(random_state=42)
    random_forest_model.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = random_forest_model.predict(X_test)
    # Evaluate the Random Forest model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Random Forest Regressor:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R-squared: {r2:.2f}")

    return random_forest_model

def save_model(model):
    # Save the model to a file
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

def __main__():
    df = load_data('Dataset\Processed-data\processed_data.csv')
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    save_model(model)

if __name__ == '__main__':
    __main__()



