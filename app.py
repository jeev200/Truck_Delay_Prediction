import os
import sys
import pandas as pd
import streamlit as st
import pickle
import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set page configuration to wide mode
st.set_page_config(layout="wide")

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
else:
    final_merge = final_merge.drop(columns=['index', 'eventtime'], errors='ignore')

# Paths from the config.ini for model, encoder, and scaler
model_dir = config['MLFLOW']['model_dir']
encoder_dir = config['MLFLOW']['encoder_dir']
scaler_dir = config['MLFLOW']['scalar_dir']

# Load model, encoder, and scaler
model = pickle.load(open(os.path.join(model_dir, "model.pkl"), 'rb'))
encoder = pickle.load(open(os.path.join(encoder_dir, "XGBoost_encoder.pkl"), 'rb'))
scaler = pickle.load(open(os.path.join(scaler_dir, "XGBoost_scaler.pkl"), 'rb'))

logging.info("Model, encoder, and scaler loaded successfully.")

# Define continuous and categorical columns
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

# Sidebar for inputs
st.sidebar.title("🚚 Truck Delay Prediction")
option = st.sidebar.selectbox('Choose a filter method:', ['Date Range', 'Truck ID', 'Route ID'])

# Initialize variables
from_date = to_date = truck_id = route_id = None

# Placeholder for the prediction results
prediction_placeholder = st.empty()
with prediction_placeholder.container():
    st.image("/app/image.jpg", width=700)  # Ensure the image path is correct
    st.write("## Predicted Delays will appear here")
    st.write("Select a filter and predict delays based on Truck ID, Route ID, or Date Range.")

if option == 'Date Range':
    from_date = pd.Timestamp(st.sidebar.date_input("Start date", value=pd.to_datetime(final_merge['departure_date']).min()), tz='UTC')
    to_date = pd.Timestamp(st.sidebar.date_input("End date", value=pd.to_datetime(final_merge['departure_date']).max()), tz='UTC')
elif option == 'Truck ID':
    truck_id = st.sidebar.selectbox('Select truck ID:', final_merge['truck_id'].unique())
elif option == 'Route ID':
    route_id = st.sidebar.selectbox('Select route ID:', final_merge['route_id'].unique())

if st.sidebar.button('Predict Delays'):
    prediction_placeholder.empty()  # Clear the placeholder before showing new data
    # Conditionally apply filters based on selected option
    filtered_data = pd.DataFrame()
    if option == 'Date Range' and from_date and to_date:
        filtered_data = final_merge[(final_merge['departure_date'] >= from_date) & (final_merge['departure_date'] <= to_date)]
    elif option == 'Truck ID' and truck_id:
        filtered_data = final_merge[final_merge['truck_id'] == truck_id]
    elif option == 'Route ID' and route_id:
        filtered_data = final_merge[final_merge['route_id'] == route_id]

    if not filtered_data.empty:
        # Reset index before further processing
        filtered_data.reset_index(drop=True, inplace=True)

        # Manual encoding using loaded encoder
        encoded_data = encoder.transform(filtered_data[encode_columns])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(), index=filtered_data.index)
        
        # Combine with unencoded categorical data
        other_cat_data = filtered_data[[col for col in cat_cols if col not in encode_columns]]
        full_cat_data = pd.concat([other_cat_data, encoded_df ], axis=1)
        
        # Scaling continuous columns using loaded scaler
        scaled_data = scaler.transform(filtered_data[cts_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=cts_cols, index=filtered_data.index)
        
        # Combine categorical and continuous data for model prediction
        final_data = pd.concat([scaled_df, full_cat_data], axis=1)
        
        predictions = model.predict(final_data)
        
        # Prepare final results to display
        results_df = filtered_data.copy()  # Copy all columns from the filtered data
        results_df['delay'] = predictions  # Append the predictions as a new column
        st.write(f"### Prediction results based on {option}:")
        st.dataframe(results_df)  # Display the results without styling
        st.success(f"{len(filtered_data)} records matched the criteria for {option}.")
    else:
        st.error("No data found for the selected option.")