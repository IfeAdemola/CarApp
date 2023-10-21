import pandas as pd
import numpy as np
import pickle

def predict(data):
    model = pickle.load(open(r'C:\Users\ifeol\projects\CarApp\model.pkl','rb'))
    return model.predict(data)
