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

def read_data(path):
    df = pd.read_csv('Dataset\Processed-data\processed_data.csv') 
    return df

def  split_data(df):
    X = df.iloc[:, :-1]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalise data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
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

