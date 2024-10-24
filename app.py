import os
import sys
import pandas as pd
import streamlit as st
import pickle
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)

# Append the parent directory to sys.path to import custom modules
parent_directory = os.path.abspath(os.path.join(__file__, "../../"))
sys.path.append(parent_directory)
logging.info(f"Added {parent_directory} to system path.")

from src.utils.config_utils import read_config
from src.utils.feature_group import fetch_feature_groups

# Reading configurations
config = read_config()
hopsworks_api_key = config['HOPSWORK']['feature_group_api']
group_names = ['final_dataset']
logging.info("Configuration read successfully.")

# Fetching feature groups
final_merge = fetch_feature_groups(hopsworks_api_key, group_names).get('final_dataset', pd.DataFrame())
if final_merge.empty:
    logging.warning("Fetched 'final_dataset' is empty.")
    st.warning("No data available to display or predict on.")

# Connect to MLflow and load artifacts
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()
model_name = "Truck_delay_Prediction"
stage = "Staging"
latest_version = client.get_latest_versions(model_name)[0]
run_id = latest_version.run_id
artifact_location = client.get_run(run_id).info.artifact_uri.replace("file://", "")
encoder_path = os.path.join(artifact_location, "encoder", "XGBoost_encoder.pkl")
scaler_path = os.path.join(artifact_location, "scaler", "XGBoost_scaler.pkl")
model_uri = f"models:/{model_name}/{stage}"
model = mlflow.pyfunc.load_model(model_uri)
encoder = pickle.load(open(encoder_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))
logging.info("Model, encoder, and scaler loaded successfully.")



cts_cols = [
        'route_avg_temp', 'route_avg_wind_speed', 'route_avg_precip', 'route_avg_humidity', 
        'route_avg_visibility', 'route_avg_pressure', 'distance', 'average_hours', 
        'temp_origin', 'wind_speed_origin', 'precip_origin', 'humidity_origin', 
        'visibility_origin', 'pressure_origin', 'temp_destination', 'wind_speed_destination', 
        'precip_destination', 'humidity_destination', 'visibility_destination', 
        'pressure_destination', 'avg_no_of_vehicles', 'truck_age', 'load_capacity_pounds', 
        'mileage_mpg', 'age', 'experience', 'average_speed_mph'
    ]
cat_cols = [
        'route_description', 'description_origin', 'description_destination', 
        'accident', 'fuel_type', 'gender', 'driving_style', 'ratings', 'is_midnight'
    ]
encode_columns = ['route_description', 'description_origin', 'description_destination', 'fuel_type', 'gender', 'driving_style']

# Streamlit interface setup
st.title('Truck Delay Prediction')
option = st.selectbox('Choose a filter method:', ['Date Range', 'Truck ID', 'Route ID'])

# Initialize variables
from_date = to_date = truck_id = route_id = None

if option == 'Date Range':
    from_date = pd.Timestamp(st.date_input("Start date", value=pd.to_datetime(final_merge['departure_date']).min()), tz='UTC')
    to_date = pd.Timestamp(st.date_input("End date", value=pd.to_datetime(final_merge['departure_date']).max()), tz='UTC')
elif option == 'Truck ID':
    truck_id = st.selectbox('Select truck ID:', final_merge['truck_id'].unique())
elif option == 'Route ID':
    route_id = st.selectbox('Select route ID:', final_merge['route_id'].unique())

if st.button('Predict Delays'):
    # Conditionally apply filters based on selected option
    filtered_data = pd.DataFrame()
    if option == 'Date Range' and from_date and to_date:
        filtered_data = final_merge[(final_merge['departure_date'] >= from_date) & (final_merge['departure_date'] <= to_date)]
    elif option == 'Truck ID' and truck_id:
        filtered_data = final_merge[final_merge['truck_id'] == truck_id]
    elif option == 'Route ID' and route_id:
        filtered_data = final_merge[final_merge['route_id'] == route_id]

    if not filtered_data.empty:
        # Manual encoding using loaded encoder
        encoded_data = encoder.transform(filtered_data[encode_columns])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(), index=filtered_data.index)
        
        # Combine with unencoded categorical data
        other_cat_data = filtered_data[[col for col in cat_cols if col not in encode_columns]]
        full_cat_data = pd.concat([encoded_df, other_cat_data], axis=1)
        
        # Scaling continuous columns using loaded scaler
        scaled_data = scaler.transform(filtered_data[cts_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=cts_cols, index=filtered_data.index)
        
        # Combine categorical and continuous data for model prediction
        final_data = pd.concat([full_cat_data, scaled_df], axis=1)
        
        # Predict using the model
        predictions = model.predict(final_data)

        # Prepare final results to display
        results_df = filtered_data[['truck_id', 'route_id']].copy()
        results_df['delay'] = predictions
        st.dataframe(results_df.style.highlight_max(axis=0))
    else:
        st.warning(f"No data found for the selected {option.lower()}.")
