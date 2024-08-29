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
crop_yield_model =joblib.load(open(f'{working_dir}/voting_yield.sav', 'rb'))

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
    # Define all subdivisions
    subdivisions = [
        'ANDAMAN & NICOBAR ISLANDS', 'ARUNACHAL PRADESH', 'ASSAM & MEGHALAYA', 'NAGA MANI MIZO TRIPURA',
        'SUB HIMALAYAN WEST BENGAL & SIKKIM', 'GANGETIC WEST BENGAL', 'ORISSA', 'JHARKHAND', 'BIHAR',
        'EAST UTTAR PRADESH', 'WEST UTTAR PRADESH', 'UTTARAKHAND', 'HARYANA DELHI & CHANDIGARH', 'PUNJAB',
        'HIMACHAL PRADESH', 'JAMMU & KASHMIR', 'WEST RAJASTHAN', 'EAST RAJASTHAN', 'WEST MADHYA PRADESH',
        'EAST MADHYA PRADESH', 'GUJARAT REGION', 'SAURASHTRA & KUTCH', 'KONKAN & GOA', 'MADHYA MAHARASHTRA',
        'MATATHWADA', 'VIDARBHA', 'CHHATTISGARH', 'COASTAL ANDHRA PRADESH', 'TELANGANA', 'RAYALSEEMA',
        'TAMIL NADU', 'COASTAL KARNATAKA', 'NORTH INTERIOR KARNATAKA', 'SOUTH INTERIOR KARNATAKA', 'KERALA',
        'LAKSHADWEEP'
    ]
    
    subdivision = st.selectbox("Subdivision", subdivisions)
    year = st.number_input("Year", min_value=1900, max_value=2100, value=2023)
    may = st.number_input("May Rainfall (mm)", min_value=0.0, value=0.0)
    jun = st.number_input("June Rainfall (mm)", min_value=0.0, value=0.0)
    jul = st.number_input("July Rainfall (mm)", min_value=0.0, value=0.0)
    aug = st.number_input("August Rainfall (mm)", min_value=0.0, value=0.0)
    sep = st.number_input("September Rainfall (mm)", min_value=0.0, value=0.0)
    
    if st.button("Predict Rainfall"):
        if subdivision and year:
            rainfall_input = {
                'SUBDIVISION': [subdivision],
                'YEAR': [year],
                'MAY': [may],
                'JUN': [jun],
                'JUL': [jul],
                'AUG': [aug],
                'SEP': [sep]
            }
            rainfall_input_df = pd.DataFrame(rainfall_input)
            
            # Predict if all inputs are provided
            if all(rainfall_input_df.iloc[0, 2:]):
                rainfall_prediction = rainfall_model.predict(rainfall_input_df)
                st.success(f"The Predicted Annual Rainfall for your Subdivision is: {rainfall_prediction[0]}")
            else:
                st.error("Please enter valid values")
        else:
            st.error("Please enter subdivision and year")


# Crop Recommendation
# Crop Recommendation
elif selected == "Crop Recommendation":
    st.title("Crop Recommendation")

    st.write("Provide the following information to get crop recommendations:")
    st.write("""
    - **Nitrogen (N)**: N-ratio in soil measured in ppm .Essential nutrient for plant growth.
    - **Phosphorus (P)**: P-ratio in ppm. Vital for root development and energy transfer.
    - **Potassium (K)**: K-ratio in pmm. Important for water regulation and disease resistance.
    - **pH Value**: Soil acidity or alkalinity level.
    - **Temperature (°C)**: Current temperature.
    - **Humidity (%)**: Moisture content in the air.
    - **Rainfall (mm)**: Amount of recent rainfall.
    """)

    N = st.number_input("Nitrogen(N) Ratio in ppm ", min_value=0, value=0)
    P = st.number_input("Phosphorus(P) Ratio in ppm", min_value=0, value=0)
    K = st.number_input("Potassium(K) Ratio in ppm", min_value=0, value=0)
    pH = st.number_input("pH Value of soil", min_value=0.0, max_value=14.0, value=0.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, value=0.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0)
    rainfall = st.number_input("Avg Rainfall (mm) in a month ", min_value=0.0, value=0.0)
    
    if st.button("Recommend Crop"):
        crop_input = np.array([[N, P, K, pH, temperature, humidity, rainfall]])
        
        if all(crop_input[0][:3]):  # Check if N, P, K values are provided
            crop_recommendation = crop_recom_model.predict(crop_input)
            st.success(f"According to Krishi Mitr you grow: {crop_recommendation[0]} in your field")
        else:
            st.error("Please enter values for Nitrogen (N), Phosphorus (P), and Potassium (K)")


# AQI Prediction
elif selected == "AQI Prediction":
    st.title("AQI Prediction")
    st.write("")
    st.write("To help predict the Air Quality Index (AQI), please provide the following information:")
    st.write("""
    - **Average Temperature (°C)**: Average temperature over a period.
    - **Maximum Temperature (°C)**: Highest temperature recorded.
    - **Minimum Temperature (°C)**: Lowest temperature recorded.
    - **Atmospheric Pressure (hPa)**: Pressure at sea level.
    - **Relative Humidity (%)**: Average percentage of humidity.
    - **Visibility (km)**: Distance at which objects are visible.
    - **Average Windspeed (km/h)**: Average wind speed.
    - **Maximum Windspeed (km/h)**: Highest wind speed recorded.
    """)

    feature_1 = st.number_input("Average Temperature (°C)", min_value=0.0, value=0.0)
    feature_2 = st.number_input("Maximum Temperature (°C)", min_value=0.0, value=0.0)
    feature_3 = st.number_input("Minimum Temperature (°C)", min_value=0.0, value=0.0)
    feature_4 = st.number_input("Atmospheric Pressure at sea level (hPa)", min_value=0.0, value=0.0)
    feature_5 = st.number_input("Average Relative Humidity (%)", min_value=0.0, value=0.0)
    feature_6 = st.number_input("Average Visibility (km)", min_value=0.0, value=0.0)
    feature_7 = st.number_input("Average Windspeed (km/h)", min_value=0.0, value=0.0)
    feature_8 = st.number_input("Maximum Windspeed (km/h)", min_value=0.0, value=0.0)
    
    if st.button("Predict AQI"):
        aqi_input = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]])
        
        if all(aqi_input[0]):  # Check if all input features are provided
            aqi_prediction = aqi_model.predict(aqi_input)
            st.success(f"Predicted AQI: {aqi_prediction[0]}")
            if(aqi_prediction[0]<25):
                st.success("This AQI is good for crops")
            elif(aqi_prediction[0]>25 && aqi_prediction[0]<=50):
                st.success("This AQI falls in the fair range")
            elif(aqi_predcition[0]>50 && aqi_prediction[0]<=100):
                st.success("Poor air quality for crops")
            elif(aqi_prediction[0]>100 && (aqi_prediction[0]<=300):
                st.success("Vey Poor air quality for crops")
             else:
                st.success("Extremly Poor Air Quality for crops")
                
        else:
            st.error("Please enter values for all AQI features")



# Meet Creators
elif selected == "Meet the Creators":
    st.title("Meet the Creators")
    st.markdown("<br>", unsafe_allow_html=True)  # Adding space between the title and the profiles

    creators = [
        {
            "name": "Kanchan Rai",
            "linkedin": "https://www.linkedin.com/in/kanchanraiii/",
            "github": "https://github.com/kanchanraiii",
            "image": "images/kanchan.jpg"
        },
        {
            "name": "Aaron Thomas",
            "linkedin": "https://www.linkedin.com/in/aaron-thomas-53996b255/",
            "github": "https://github.com/AayJayTee",
            "image": "images/aaron.jpg"
        },
        {
            "name": "Saumyaa Garg",
            "linkedin": "https://www.linkedin.com/in/saumyaa-garg-481b9224b/",
            "github": "https://github.com/saumyaagarg",
            "image": "images/saumyaa.jpg"
        }
    ]

    cols = st.columns(3)

    for i, creator in enumerate(creators):
        with cols[i]:
            st.image(creator["image"], width=80, caption=None, use_column_width=True, output_format='auto')
            st.markdown(f"<div class='profile-column'><p class='profile-name'>{creator['name']}</p><a href='{creator['linkedin']}'><img src='https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg' class='icon'></a> <a href='{creator['github']}'><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' class='icon'></a></div>", unsafe_allow_html=True)

# Crop Yield Prediction
elif selected == "Crop Yield Prediction":
    st.title("Crop Yield Prediction")
    st.write("")
    st.markdown("""
### Using the Crop Yield Prediction Model

- **Select State**: Choose the state where the crop is being cultivated.
- **Select Crop**: Pick the specific crop for yield prediction.
- **Select Season**: Choose the appropriate growing season.
- **Input Soil pH**: Enter the soil pH level. [Measure pH at home](https://www.youtube.com/watch?v=mZgxUqoJMcg).
- **Input Rainfall**: Enter the rainfall amount (mm). [Check local rainfall](https://mausam.imd.gov.in/responsive/rainfallinformation.php).
- **Input Temperature**: Enter the average temperature (°C). [Check local temperature](https://www.accuweather.com/).
- **Input Area**: Enter the cultivation area (hectares).
- **Input Production**: Enter the total production (tons).
- **Click Predict**: Get the yield prediction.
### Interpreting Results

- Predicted yield is shown in tons per hectare.
- Use this data for crop management and planning.
                
- Production input in tonnes denotes the quantity of a particular crop that farmers plan to cultivate. This metric, measured in metric tons, represents the amount of crops like wheat, corn, or rice that farmers intend to sow and grow during a specific agricultural period.                

Leverage machine learning for accurate crop yield predictions to enhance productivity and sustainability.
""")

    states = ['Andaman and Nicobar Islands', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh', 
              'Dadra and Nagar Haveli', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 
              'Jammu and Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 
              'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 
              'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 
              'Uttarakhand', 'West Bengal']

    crops = ['Arecanut', 'Barley', 'Banana', 'Blackpepper', 'Brinjal', 'Cabbage', 'Cardamom', 'Cashewnuts', 'Cauliflower', 
             'Coriander', 'Cotton', 'Garlic', 'Grapes', 'Horsegram', 'Jowar', 'Jute', 'Ladyfinger', 'Maize', 
             'Mango', 'Moong', 'Onion', 'Orange', 'Papaya', 'Pineapple', 'Potato', 'Rapeseed', 'Ragi', 'Rice', 
             'Sesamum', 'Soyabean', 'Sunflower', 'Sweetpotato', 'Tapioca', 'Tomato', 'Turmeric', 'Wheat']

    seasons = ['Kharif', 'Rabi', 'Summer', 'Whole Year']

    state = st.selectbox("Select State", states)
    crop = st.selectbox("Select Crop", crops)
    season = st.selectbox("Select Season", seasons)
    pH = st.number_input("Soil pH Value", min_value=0.0, max_value=14.0, value=0.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=0.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, value=0.0)
    area = st.number_input("Area (hectares)", min_value=0.0, value=0.0)
    production = st.number_input("Production (tons)", min_value=0.0, value=0.0)

    if st.button("Predict Yield"):
        if state and crop and season and pH and rainfall and temperature and area and production:
            state_lower = state.lower()
            crop_lower = crop.lower()
            season_lower = season.lower()

            state_encoded = [0] * (len(states) - 1) if state_lower == 'andaman and nicobar islands' else [1 if s.lower() == state_lower else 0 for s in states if s.lower() != 'andaman and nicobar islands']
            crop_encoded = [0] * (len(crops) - 1) if crop_lower == 'arecanut' else [1 if c.lower() == crop_lower else 0 for c in crops if c.lower() != 'arecanut']
            season_encoded = [0] * (len(seasons) - 1) if season_lower == 'kharif' else [1 if s.lower() == season_lower else 0 for s in seasons if s.lower() != 'kharif']

            input_features = np.array(state_encoded + crop_encoded + season_encoded + [pH, rainfall, temperature, area, production]).reshape(1, -1)

            expected_num_features = len(states) + len(crops) + len(seasons) - 3 + 5

            if input_features.shape[1] != expected_num_features:
                st.error(f"Feature shape mismatch, expected: {expected_num_features}, got: {input_features.shape[1]}")
            else:
                predicted_yield = crop_yield_model.predict(input_features)
                st.success(f'The predicted yield for the selected inputs is: {predicted_yield[0]:.2f} tons/hectare')
        else:
            st.error("Please enter all required values")


# Home
else:
    img = "hero2.jpg"
    st.title("Krishi Mitr")
    st.write("##### Welcome to Krishi Mitr! Explore our tools in the sidebar to make informed agricultural decisions.")
    st.image(img, width=750)
    
    st.write("")  # Leave some space after the image
    st.write("### Overview")  # Section title for introduction
    st.write("Krishi Mitr is designed to empower farmers with advanced predictive tools. Predict rainfall accurately with our Ridge Regression model. Get personalized crop recommendations using our Random Forest algorithm tailored to your local conditions. Estimate crop yields confidently with a combination of CatBoost, XGBoost, and Decision Tree models. Assess Air Quality Index (AQI) with precision using our XGBoost model, fine-tuned with Grid Search CV. Harness these features to optimize farming strategies and increase productivity.")
    st.write("### Find the Code at:")
    st.write("Link: https://github.com/kanchanraiii/Krishi-Mitr")
