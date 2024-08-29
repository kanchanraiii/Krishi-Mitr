import streamlit as st
import pickle
import numpy as np
import os
import joblib
import pandas as pd
from streamlit_option_menu import option_menu

# Loading all the models
working_dir = os.path.dirname(os.path.abspath(__file__))
crop_recom_model = pickle.load(open(f'{working_dir}/RF_Crop.sav', 'rb'))
rainfall_model = pickle.load(open(f'{working_dir}/Rainfall_Ridge.sav', 'rb'))
aqi_model = joblib.load(f'{working_dir}/xgb_best_model.joblib', 'rb')
crop_yield_model = joblib.load(open(f'{working_dir}/voting_yield.sav', 'rb'))

st.set_page_config(
    page_title="Krishi Mitr",
    page_icon=":corn:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set background color
st.markdown(
    """
    <style>
        body {
            background-color: #f0f5f5;
        }
        .profile-pic {
            border-radius: 50%;
            width: 150px;
            height: 150px;
            object-fit: cover;
            margin-bottom: 10px;
        }
        .profile-column {
            text-align: center;
            padding: 20px;
        }
        .icon {
            width: 24px;
            height: 24px;
            margin: 0 5px;
        }
        .profile-name {
            font-size: 18px;
            font-weight: bold;
            margin-top: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    options = ["Home", "Rainfall Prediction", "Crop Yield Prediction", "Crop Recommendation", "AQI Prediction", "Meet the Creators"]
    selected = option_menu("Krishi Mitr",
                           options,
                           menu_icon=":seedling:",
                           icons=["house", "cloud-rain", "tree", "tree", "wind", "people"],
                           default_index=0)

# Rainfall Prediction
if selected == "Rainfall Prediction":
    st.title("Rainfall Prediction")
    st.write("Provide the following information to predict annual rainfall:")
    st.write("""
    - **Subdivision**: Select your geographical area.
    - **Year**: Enter the year for which you wish to predict rainfall.
    - **May Rainfall (mm)**: Amount of rainfall in May.
    - **June Rainfall (mm)**: Amount of rainfall in June.
    - **July Rainfall (mm)**: Amount of rainfall in July.
    - **August Rainfall (mm)**: Amount of rainfall in August.
    - **September Rainfall (mm)**: Amount of rainfall in September.
    """)
    
    subdivisions = [
