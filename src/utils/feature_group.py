import time
import hopsworks
import pandas as pd
import traceback
import hsfs

feature_descriptions = {
    "city_weather": [
        {"name": "index", "description": "unique identification for each weather record"},
        {"name": "city_id", "description": "unique identification for each city"},
        {"name": "datetime", "description": "date of the weather observation"},
        {"name": "temp", "description": "temperature at the time of the weather observation, in Fahrenheit"},
        {"name": "wind_speed", "description": "wind speed during the observation, in miles per hour"},
        {"name": "description", "description": "brief description of the weather condition (e.g., Sunny, Cloudy)"},
        {"name": "precip", "description": "precipitation level during the observation, in inches"},
        {"name": "humidity", "description": "humidity percentage during the observation"},
        {"name": "visibility", "description": "visibility in miles at the time of the observation"},
        {"name": "pressure", "description": "atmospheric pressure at the time of the observation, in millibars"},
        {"name": "eventtime", "description": "dummy event time for this weather record"}
    ],
    "drivers_table": [
        {"name": "driver_id", "description": "unique identification for each driver"},
        {"name": "name", "description": "name of the truck driver"},
        {"name": "gender", "description": "gender of the truck driver"},
        {"name": "age", "description": "age of the truck driver"},
        {"name": "experience", "description": "experience of the truck driver in years"},
        {"name": "driving_style", "description": "driving style of the truck driver, conservative or proactive"},
        {"name": "ratings", "description": "average rating of the truck driver on a scale of 1 to 10"},
        {"name": "vehicle_no", "description": "the number of the driver’s truck"},
        {"name": "average_speed_mph", "description": "average speed of the truck driver in miles per hour"},
        {"name": "eventtime", "description": "dummy event time"},
        {"name": "index", "description": "unique identification for each weather record"},
    ],
    "trucks_table": [
        {"name": "index", "description": "unique identification for each truck record"},
        {"name": "truck_id", "description": "unique identification for each truck"},
        {"name": "truck_age", "description": "age of the truck in years"},
        {"name": "load_capacity_pounds", "description": "maximum load capacity of the truck in pounds (some values may be missing)"},
        {"name": "mileage_mpg", "description": "truck's fuel efficiency measured in miles per gallon"},
        {"name": "fuel_type", "description": "type of fuel used by the truck (e.g., gas, diesel)"},
        {"name": "eventtime", "description": "the timestamp when the event or record was created"}
    ],
    "routes_table": [
        {"name": "index", "description": "unique identification for each route record"},
        {"name": "route_id", "description": "unique identification for each route"},
        {"name": "origin_id", "description": "unique identification for the origin city or location"},
        {"name": "destination_id", "description": "unique identification for the destination city or location"},
        {"name": "distance", "description": "distance between origin and destination in miles"},
        {"name": "average_hours", "description": "average travel time between origin and destination in hours"},
        {"name": "eventtime", "description": "the timestamp when the event or record was created"}
    ],
    "traffic_table": [
        {"name": "index", "description": "unique identification for each route activity record"},
        {"name": "route_id", "description": "unique identification for each route"},
        {"name": "datetime", "description": "date of the route activity"},
        {"name": "no_of_vehicles", "description": "number of vehicles on the route during the recorded hour"},
        {"name": "accident", "description": "whether an accident occurred (0 for no accident, 1 for accident)"},
        {"name": "eventtime", "description": "the timestamp when the event or record was created"}
    ],
    "truck_schedule_table": [
        {"name": "index", "description": "unique identification for each truck schedule record"},
        {"name": "truck_id", "description": "unique identification for each truck"},
        {"name": "route_id", "description": "unique identification for each route"},
        {"name": "departure_date", "description": "the departure date and time of the truck"},
        {"name": "estimated_arrival", "description": "the estimated arrival date and time of the truck"},
        {"name": "delay", "description": "whether the truck was delayed (0 for no delay, 1 for delayed)"},
        {"name": "eventtime", "description": "the timestamp when the event or record was created"}
    ],
    "routes_weather": [
        {"name": "index", "description": "unique identification for each weather record on the route"},
        {"name": "route_id", "description": "unique identification for each route"},
        {"name": "date", "description": "date and time of the weather observation"},
        {"name": "temp", "description": "temperature at the time of the weather observation, in Fahrenheit"},
        {"name": "wind_speed", "description": "wind speed during the observation, in miles per hour"},
        {"name": "description", "description": "brief description of the weather condition (e.g., Sunny, Rain Shower)"},
        {"name": "precip", "description": "precipitation level during the observation, in inches"},
        {"name": "humidity", "description": "humidity percentage during the observation"},
        {"name": "visibility", "description": "visibility in miles at the time of the observation"},
        {"name": "pressure", "description": "atmospheric pressure at the time of the observation, in millibars"},
        {"name": "eventtime", "description": "the timestamp when the event or record was created"}
    ]
}

def create_or_update_feature_groups_in_hopsworks(hopsworks_api_key, cleaned_data, version=1, retries=3, delay=60):
    try:
        # Log into Hopsworks and get the feature store
        project = hopsworks.login(api_key_value=hopsworks_api_key)
        fs = project.get_feature_store()

        for table_name, df in cleaned_data.items():
            # Ensure 'eventtime' column is mapped correctly, assuming it already exists in the DataFrame
            if 'eventtime' in df.columns and pd.api.types.is_datetime64_any_dtype(df['eventtime']):
                df['eventtime'] = df['eventtime'].dt.date

            attempts = 0
            while attempts < retries:
                try:
                    # Try to get the existing feature group
                    feature_group = fs.get_feature_group(name=table_name, version=version)
                    print(f"Feature group '{table_name}' (version {version}) already exists.")

                    break  # Stop retrying after successful insert

                except Exception as e:
                    # Handle parallel execution limit error
                    if "Parallel executions quota reached" in str(e) or "500" in str(e):
                        time.sleep(delay)  # Wait for the specified delay before retrying
                        attempts += 1
                        print(f"Retry {attempts}/{retries} for {table_name} due to parallel execution limits.")
                    elif "Could not find" in str(e) or "Not Found" in str(e):
                        # If the feature group does not exist, create it
                        if attempts == 0:  # Ensure creation is only attempted once
                            primary_key = ['index']  # Assuming 'index' is available as the primary key
                            feature_group = fs.create_feature_group(
                                name=table_name,
                                version=version,
                                primary_key=primary_key,
                                event_time='eventtime',  # Map the existing 'eventtime' column as the event time
                                description=f"Created feature group for {table_name}.",
                                online_enabled=False  # Adjust based on your requirements
                            )
                            feature_group.save(df)  # Save the data for the first time
                            print(f"New feature group '{table_name}' (version {version}) created and data saved.")

                            # Update feature descriptions from the global dictionary (only when created)
                            if table_name in feature_descriptions:
                                for desc in feature_descriptions[table_name]:
                                    feature_group.update_feature_description(desc["name"], desc["description"])
                                print(f"Feature descriptions updated for '{table_name}'.")

                            # Configure and compute statistics for the feature group
                            feature_group.statistics_config = {
                                "enabled": True,        # Enable statistics calculation
                                "histograms": True,     # Include histograms in the statistics
                                "correlations": True    # Include correlations in the statistics
                            }
                            feature_group.update_statistics_config()  # Update the statistics config
                            # feature_group.compute_statistics()        # Compute the statistics
                            print(f"Statistics computed for feature group '{table_name}'.")

                            break
                        else:
                            print(f"Not retrying creation for '{table_name}', feature group not found.")
                            break
                    else:
                        # For any other exceptions, do not retry and raise the error
                        print(f"Unexpected error for '{table_name}': {e}")
                        break
                if attempts >= retries:
                    print(f"Failed to create feature group '{table_name}' after {retries} attempts due to parallel execution limits.")
                    break

    except Exception as e:
        print(f"General error during feature group management: {e}")
        raise

# def create_or_update_feature_groups_in_hopsworks(hopsworks_api_key, cleaned_data, version=1, retries=3, delay=60):
#     try:
#         # Log into Hopsworks and get the feature store
#         project = hopsworks.login(api_key_value=hopsworks_api_key)
#         fs = project.get_feature_store()

#         for table_name, df in cleaned_data.items():
#             if 'eventtime' in df.columns and pd.api.types.is_datetime64_any_dtype(df['eventtime']):
#                 df['eventtime'] = df['eventtime'].dt.date

#             attempts = 0
#             while attempts < retries:
#                 try:
#                     # Try to get the existing feature group
#                     feature_group = fs.get_feature_group(name=table_name, version=version)
#                     print(f"Feature group '{table_name}' (version {version}) already exists.")
                    
#                     # Insert new data into the existing feature group
#                     feature_group.insert(df)
#                     print(f"Data inserted into existing feature group '{table_name}' (version {version}).")
#                     break  # Stop retrying after successful insert

#                 except Exception as e:
#                     # Log the error and retry if it's a parallel execution limit error
#                     if "Parallel executions quota reached" in str(e) or "500" in str(e):
#                         attempts += 1
#                         delay_time = delay * (2 ** (attempts - 1))  # Exponential backoff
#                         print(f"Retry {attempts}/{retries} for {table_name} due to parallel execution limits. Waiting for {delay_time} seconds.")
#                         time.sleep(delay_time)  # Wait before retrying
#                     elif "Could not find" in str(e) or "Not Found" in str(e):
#                         if attempts == 0:  # Attempt creation only once
#                             # Creating new feature group
#                             primary_key = ['index']  # Assuming 'index' is the primary key
#                             feature_group = fs.create_feature_group(
#                                 name=table_name,
#                                 version=version,
#                                 primary_key=primary_key,
#                                 event_time='eventtime',
#                                 description=f"Created feature group for {table_name}.",
#                                 online_enabled=False
#                             )
#                             feature_group.save(df)
#                             print(f"New feature group '{table_name}' created and data saved.")
#                             break
#                     else:
#                         print(f"Unexpected error for '{table_name}': {e}")
#                         break
#                 if attempts >= retries:
#                     print(f"Failed to create feature group '{table_name}' after {retries} attempts.")
#                     break

#     except Exception as e:
#         print(f"General error during feature group management: {e}")
#         raise




# def delete_feature_groups(hopsworks_api_key, feature_group_names, version=1):
#     try:
#         project = hopsworks.login(api_key_value=hopsworks_api_key)
#         fs = project.get_feature_store()

#         for name in feature_group_names:
#             try:
#                 feature_group = fs.get_feature_group(name, version=version)
#                 feature_group.delete()
#                 print(f"Deleted feature group: {name} version {version}")
#             except Exception as e:
#                 print(f"Error deleting feature group {name} version {version}: {e}")
#     except Exception as e:
#         print(f"Error during feature group deletion process: {e}")
#         raise
def delete_feature_groups(hopsworks_api_key, feature_group_names, version=1, retry_attempts=3):
    try:
        # Login to Hopsworks using the API key
        project = hopsworks.login(api_key_value=hopsworks_api_key)
        fs = project.get_feature_store()

        for name in feature_group_names:
            attempt = 0
            while attempt < retry_attempts:
                try:
                    # Try to get and delete the feature group
                    feature_group = fs.get_feature_group(name, version=version)
                    feature_group.delete()
                    print(f"Deleted feature group: {name} version {version}")
                    break  # Exit loop if deletion is successful
                except Exception as e:
                    attempt += 1
                    print(f"Error deleting feature group {name} version {version} on attempt {attempt}: {e}")
                    if attempt >= retry_attempts:
                        print(f"Failed to delete feature group {name} after {retry_attempts} attempts.")
                    else:
                        print("Retrying in 60 seconds...")
                        time.sleep(60)  # Wait for 5 seconds before retrying
    except Exception as e:
        print(f"Error during feature group deletion process: {e}")
        raise


# def fetch_feature_groups(hopsworks_api_key, group_names, version=1):
#     try:
#         project = hopsworks.login(api_key_value=hopsworks_api_key)
#         fs = project.get_feature_store()
#         feature_groups = {}

#         for name in group_names:
#             feature_group = fs.get_feature_group(name, version=version)
#             df = feature_group.read()
#             feature_groups[name] = df
#             print(f"Feature group '{name}' version {version} fetched successfully.")

#         return feature_groups
#     except Exception as e:
#         print(f"Error fetching feature groups: {e}")
#         raise

def fetch_feature_groups(hopsworks_api_key, group_names, version=1, retries=3, delay=60):
    try:
        project = hopsworks.login(api_key_value=hopsworks_api_key)
        print("Logged into Hopsworks successfully.")
        fs = project.get_feature_store()

        feature_groups = {}

        for name in group_names:
            attempts = 0
            while attempts < retries:
                try:
                    feature_group = fs.get_feature_group(name=name, version=version)
                    df = feature_group.read()
                    feature_groups[name] = df
                    print(f"Feature group '{name}' version {version} fetched successfully.")
                    break  # Stop retrying after a successful fetch
                except hsfs.client.exceptions.RestAPIError as e:
                    if "ConcurrentAccessTimeoutException" in str(e):
                        attempts += 1
                        print(f"Retry {attempts}/{retries} due to lock timeout for feature group '{name}'. Waiting {delay} seconds...")
                        time.sleep(delay)  # Wait before retrying
                    else:
                        print(f"Error fetching feature group '{name}': {e}")
                        traceback.print_exc()
                        break
        return feature_groups
    except Exception as e:
        print(f"General error fetching feature groups: {e}")
        traceback.print_exc()
        raise