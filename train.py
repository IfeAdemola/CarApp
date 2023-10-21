import pandas as pd
import joblib
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocessing import train_preprocessor

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
    

def save_model(model, file_path):
    """
    Save a machine learning model to a file using joblib.

    Args:
        model: The trained machine learning model to be saved.
        file_path (str): The file path where the model will be saved.
    """
    try:
        joblib.dump(model, file_path)
        print(f"Model saved to {file_path}")
    except Exception as e:
        print(f"Error while saving the model: {str(e)}")

def load_model(file_path):
    """
    Load a machine learning model from a file using joblib.

    Args:
        file_path (str): The file path from which to load the model.

    Returns:
        model: The loaded machine learning model.
    """
    try:
        model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")
        return model
    except Exception as e:
        print(f"Error while loading the model: {str(e)}")
        return None



def main(data_path, model_path, ct_path):
    df = load_data(data_path)
    output = train_preprocessor(df)
    model = train_model(*output['data'])
    save_model(model, model_path)
    save_model(output['ct'], ct_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process data from a specified path")
    parser.add_argument("--data_path", type=str, help="Path to the data file")
    parser.add_argument("--model_path", type=str, help="Path to the model file")
    parser.add_argument("--ct_path", type=str, help="Path to the transformer file")

    args = parser.parse_args()
    main(args.data_path, args.model_path, args.ct_path)



