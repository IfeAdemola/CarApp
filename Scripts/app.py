import streamlit as st
import numpy as np
from inference import predict

st.title('Car App') 
st.markdown('Predicting the price of a car based on the attributes of the car')

st.sidebar.markdown("## Variable Selector")
volume = st.sidebar.number_input('Volume of car')
curbweight = st.sidebar.number_input('Curbweight of car')
peakrpm = st.sidebar.number_input('Peakrpm of car')
citympg = st.sidebar.number_input('Citympg of car')
horsepower = st.sidebar.number_input('Horsepower of car')

fueltype = st.sidebar.selectbox(
"What type of fuel does the car use?",
("Diesel", "Gas"),
index=None,
placeholder="Select fuel type...",
)

if fueltype == "Diesel":
    fueltype_diesel = 1
    fueltype_gas = 0
else:
    fueltype_diesel = 0
    fueltype_gas = 1

if st.button('Predict car price'):
    data = np.array([[volume, curbweight, peakrpm, citympg, horsepower, fueltype_diesel, fueltype_gas]])
    price = predict(data)
    st.write(price[0])
          





