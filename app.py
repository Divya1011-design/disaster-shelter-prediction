import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model & preprocessor
model = joblib.load("shelter_model_rf.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# App layout
st.set_page_config(page_title="Shelter Demand Predictor", page_icon="🏠", layout="centered")

# Header
st.title("🏠 Disaster Shelter Demand Prediction")
st.markdown(
    """
    This tool predicts the **estimated number of people affected (shelter demand)** 
    during a disaster based on historical disaster data.  
    👉 Fill in details in the sidebar, then click **Predict**.
    """
)

# Sidebar inputs
st.sidebar.header("📌 Disaster Details")

year = st.sidebar.number_input("Year", min_value=1900, max_value=2100, value=2025)
country = st.sidebar.text_input("Country", "India")

disaster_group = st.sidebar.selectbox("Disaster Group", ["Natural", "Technological"])
disaster_subgroup = st.sidebar.selectbox("Disaster Subgroup", 
    ["Climatological", "Hydrological", "Geophysical", "Meteorological"])
disaster_type = st.sidebar.selectbox("Disaster Type", 
    ["Flood", "Storm", "Earthquake", "Drought", "Epidemic"])
disaster_subtype = st.sidebar.text_input("Disaster Subtype", "Riverine flood")

total_events = st.sidebar.number_input("Total Events", min_value=1, value=1)
total_deaths = st.sidebar.number_input("Total Deaths", min_value=0, value=0)
total_damage = st.sidebar.number_input("Total Damage (USD, Adjusted)", min_value=0, value=0)

# Main screen - prediction
st.markdown("###  Predict Shelter Demand")

if st.button("📊 Predict"):
    # Validate inputs (basic check)
    if total_events <= 0:
        st.error("⚠️ Total events must be greater than 0.")
    else:
        # Prepare input
        input_data = pd.DataFrame([{
            "Year": year,
            "Country": country,
            "Disaster_Group": disaster_group,
            "Disaster_Subroup": disaster_subgroup,
            "Disaster_Type": disaster_type,
            "Disaster_Subtype": disaster_subtype,
            "Total_Events": total_events,
            "Total_Deaths": total_deaths,
            "Total_Damage_USD_adjusted": total_damage
        }])

        # Predict
        X_trans = preprocessor.transform(input_data)
        pred_log = model.predict(X_trans)
        pred = np.expm1(pred_log)

        # Show result
        st.markdown(
            f"""
            <div style="padding:20px; background-color:#f9f9f9; border:1px solid #ddd; 
                        border-radius:10px; text-align:center;">
                <h3 style="color:#444;">Estimated People Affected</h3>
                <h2 style="color:#d9534f;">{int(pred[0]):,}</h2>
                <p style="color:#666;">(Approximate shelter demand prediction)</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Footer
st.markdown("---")
st.caption("Developed using Machine Learning & Streamlit | For planning and awareness purposes only")





