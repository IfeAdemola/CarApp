import logging
import streamlit as st
from app import (load_model,
                 predict, 
                 inference_preprocessor)

# Configure the logger
logging.basicConfig(level=logging.INFO)
# Create a logger instance
logger = logging.getLogger("MyConsoleLogger")
# Create a StreamHandler and add it to the logger
# stream handler will send your logs to the terminal
# We call the terminal standard output professionally
# You can send logs to a file, for that you create, you guessed it, a filehandler
console_handler = logging.StreamHandler()
# Create a formatter for your stream handler
# basically sets the format the log output should look like
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
# add the handler to the logger
logger.addHandler(console_handler)

model_path = 'artifacts/models/model.joblib'
preprocessor_path = 'artifacts/models/ct.joblib'

@st.cache_resource
def load_models(model_path, preprocessor_path):  
    model = load_model(model_path)
    preprocessor = load_model(preprocessor_path)
    return model, preprocessor

def main():    
    st.title('Car App') 
    st.markdown('Predicting the price of a car based on the attributes of the car')
    st.sidebar.markdown("## Variable Selector")

    horsepower = st.sidebar.number_input('Horsepower of car')
    fueltype = st.sidebar.selectbox(
    "What type of fuel does the car use?",
    ("diesel", "gas"),
    index=None,
    placeholder="Select fuel type...",
    )

    model, preprocessor = load_models(model_path, preprocessor_path)

    if st.button('Predict car price'):
        price = run(model, preprocessor,horsepower, fueltype)
        st.markdown(f"**The price of your car is ${price:.2f}**")


def run(model, preprocessor, horsepower, fueltype):
    # put the prediction in a try catch block to handle errors safely
    try:
        data = inference_preprocessor(fueltype, float(horsepower), preprocessor)
        price = predict(model, data)
        logger.info(f"Prediction was ${price}")
    except Exception as e:
        logger.error(f"An error occurred: {e} ")
    return price


if __name__ == "__main__":
    main()


