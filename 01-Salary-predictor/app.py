import streamlit as st  
import pickle
import pandas as pd
import os

# Set browser tab tittle & icon
st.set_page_config(page_title = "Salary Predictor", page_icon="💸")  


# Load the model 
@st.cache_resource  # Optional: to avoid reloading every time
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model



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
    input = pd.DataFrame({'YearsExperience': [years_exp]})
    prediction= model.predict(input)
    st.success(f"Predicted Salary:  {prediction[0]:,.2f}")