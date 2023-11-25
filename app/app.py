import streamlit as st
from scripts.preprocessing import inference_preprocessor
from scripts.inference import load_model, predict

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
    
    data = inference_preprocessor(fueltype, float(horsepower), preprocessor)
    price = predict(model, data)
    return price


if __name__ == "__main__":
    main()


