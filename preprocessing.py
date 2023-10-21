import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split

features = ['fueltype', 'volume', 'curbweight', 'peakrpm', 'citympg', 'horsepower']
target = ['price']

def train_preprocessor(data):
    data['volume'] = data.apply(lambda row: row['carlength'] * row['carheight'] * row['carwidth'], axis=1)
    
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ct = make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include=np.number)),  # rating
        (OneHotEncoder(), make_column_selector(dtype_include=object)))  # city
    X_train = ct.fit_transform(X_train)

    return {
        'preprocessor': ct,
        'data': (X_train, X_test, y_train, y_test)
        }


def inference_preprocessor(fueltype, horsepower, ct):
    curbweight = np.random.uniform(1488, 4066)
    peakrpm = np.random.uniform(4150, 6600)
    citympg = np.random.uniform(13, 49)
    carlength = np.random.uniform(141.1, 208.1)
    carheight = np.random.uniform(47.8, 59.8)
    carwidth = np.random.uniform(60.3, 72.3)
    volume = carlength * carheight * carwidth
   
    data={
        'fueltype': fueltype,
        'volume': volume,
        'curbweight': curbweight,
        'peakrpm': peakrpm,
        'citympg': citympg,
        'horsepower': horsepower}
    data = pd.DataFrame([data])
    
    data = ct.transform(data)
    return data


