import streamlit as st  
import pickle
import numpy as np


# Set browser tab tittle & icon
st.set_page_config(page_title = "Salary Predictor", page_icon="💸")  


# Load the model 
@st.cache_resource  # Optional: to avoid reloading every time
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)
    

model = load_model()   

# Streamlit UI
st.title("📈 Salary Predictor")  

# Get user input 
years_exp = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=1.5,
    step=0.5)


if st.button("Predict Salary"):
    input_data = np.array([[years_exp]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Salary:  {prediction[0]:,.2f}")