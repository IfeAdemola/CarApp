import pandas as pd
import numpy as np
import joblib
import argparse

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
