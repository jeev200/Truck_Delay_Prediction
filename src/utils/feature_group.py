import time
import hopsworks
import pandas as pd

def create_or_update_feature_groups_in_hopsworks(hopsworks_api_key, cleaned_data, version=1, retries=3, delay=60):
    try:
        # Log into Hopsworks and get the feature store
        project = hopsworks.login(api_key_value=hopsworks_api_key)
        fs = project.get_feature_store()

        for table_name, df in cleaned_data.items():
            # Convert 'eventtime' column to date format if it exists
            if 'eventtime' in df.columns and pd.api.types.is_datetime64_any_dtype(df['eventtime']):
                df['eventtime'] = df['eventtime'].dt.date

            attempts = 0
            while attempts < retries:
                try:
                    # Try to get the existing feature group
                    feature_group = fs.get_feature_group(name=table_name, version=version)
                    print(f"Feature group '{table_name}' (version {version}) already exists.")
                    break  # If it exists, print message and stop trying
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
                                description=f"created feature group for {table_name}.",
                                online_enabled=False  # Adjust based on your requirements
                            )
                            feature_group.save(df)
                            print(f"New feature group '{table_name}' (version {version}) created.")
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



# def create_or_update_feature_groups_in_hopsworks(hopsworks_api_key, cleaned_data, version=1):
#     try:
#         project = hopsworks.login(api_key_value=hopsworks_api_key)
#         fs = project.get_feature_store()

#         for table_name, df in cleaned_data.items():
#             if 'eventtime' in df.columns and pd.api.types.is_datetime64_any_dtype(df['eventtime']):
#                 df['eventtime'] = df['eventtime'].dt.date

#                 try:
#                     feature_group = fs.get_feature_group(name=table_name, version=version)
#                     feature_group.insert(df, write_options={"upsert": True})
#                     print(f"Updating feature group '{table_name}' version {version} with new data.")
#                 except Exception as e:
#                     feature_group = fs.create_feature_group(
#                         name=table_name,
#                         version=version,
#                         primary_key=['index'],
#                         description=f"Features for {table_name}",
#                         event_time='eventtime',
#                         online_enabled=False
#                     )
#                     feature_group.insert(df)
#                     print(f"Created new feature group '{table_name}' version {version}.")
#     except Exception as e:
#         print(f"Error during feature group management: {e}")
#         raise

def delete_feature_groups(hopsworks_api_key, feature_group_names, version=1):
    try:
        project = hopsworks.login(api_key_value=hopsworks_api_key)
        fs = project.get_feature_store()

        for name in feature_group_names:
            try:
                feature_group = fs.get_feature_group(name, version=version)
                feature_group.delete()
                print(f"Deleted feature group: {name} version {version}")
            except Exception as e:
                print(f"Error deleting feature group {name} version {version}: {e}")
    except Exception as e:
        print(f"Error during feature group deletion process: {e}")
        raise


def fetch_feature_groups(hopsworks_api_key, group_names, version=1):
    try:
        project = hopsworks.login(api_key_value=hopsworks_api_key)
        fs = project.get_feature_store()
        feature_groups = {}

        for name in group_names:
            feature_group = fs.get_feature_group(name, version=version)
            df = feature_group.read()
            feature_groups[name] = df
            print(f"Feature group '{name}' version {version} fetched successfully.")

        return feature_groups
    except Exception as e:
        print(f"Error fetching feature groups: {e}")
        raise
