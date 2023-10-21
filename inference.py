import pandas as pd
import numpy as np
import joblib
import argparse
from preprocessing import inference_preprocessor

def predict(model, data):
    prediction = model.predict(data)
    prediction = float(prediction)
    return prediction

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
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process data from a specified path")
    parser.add_argument("--model_path", type=str, help="Path to the model file")
    parser.add_argument("--ct_path", type=str, help="Path to the transformer file")
    parser.add_argument("--fueltype", type=str, help="Fueltype")
    parser.add_argument("--horsepower", type=float, help="Horsepower")
    args = parser.parse_args()
    model = load_model(args.model_path)
    preprocessor = load_model(args.ct_path)

    fueltype = args.fueltype
    horsepower = float(args.horsepower)
    data = inference_preprocessor(fueltype, horsepower, preprocessor)

    prediction = predict(model, data)
    print(f"The price of the car is ${prediction:.2f} USD")

