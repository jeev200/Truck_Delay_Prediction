import configparser
import os
import sys
import hopsworks
import traceback
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_directory)

from src.utils.config_utils import *
from src.utils.db_utlis import * 
from src.utils.feature_group import *

from src.components.data_cleaning import DataCleaner



STAGE_NAME = "Data Cleaning"

class DataIngestionAndCleaning:
    def __init__(self):
        self.config = read_config()
        self.db_config = get_db_config(self.config)
        self.hopsworks_api_key = self.config.get('HOPSWORK', 'feature_group_api')

    def main(self):
        engine = connect_to_db(self.db_config)
        cleaner = DataCleaner(hopsworks_api_key=self.hopsworks_api_key)
        
        cleaned_data = fetch_all_data(engine)
        for table_name, df in cleaned_data.items():
            df = cleaner.remove_duplicates(df)
            
            
            # Handle negative experiences if the table contains 'experience' column
            if 'experience' in df.columns:
                df = cleaner.exp(df)
            df = cleaner.remove_outliers(df, table_name)
            df = cleaner.filling_missing_values(df)
            df = cleaner.convert_string_to_datetime(df, table_name)
            df = cleaner.add_index_and_eventtime_columns(df)

            if table_name == 'truck_schedule_table' and 'estimated_arrival' in df.columns:
                df = cleaner.convert_estimated_arrival(df, 'estimated_arrival')
            
            if 'date' in df.columns and 'hour' in df.columns:
                df = cleaner.combine_date_hour(df, 'date', 'hour')
            
            columns_to_drop = {
                'city_weather': ['chanceofrain', 'chanceoffog', 'chanceofsnow', 'chanceofthunder', 'date', 'hour'],
                'routes_weather': ['chanceofrain', 'chanceoffog', 'chanceofsnow', 'chanceofthunder'],
                'traffic_table': ['date', 'hour']
            }
            if table_name in columns_to_drop:
                df = cleaner.drop_columns(df, columns_to_drop[table_name])
            
            print(f"Data types for table '{table_name}':")
            print(df.dtypes)

            cleaned_data[table_name] = df

        # replace_cleaned_data(cleaned_data, engine)
        
        close_db_connection(engine)
        
        version = 1 
        
        # # Call the delete function
        # feature_group_names = [
        #     'city_weather', 'routes_weather', 'traffic_table',
        #     'truck_schedule_table', 'trucks_table', 'drivers_table', 'routes_table'
        # ]
        # delete_feature_groups(self.hopsworks_api_key, feature_group_names, version)
        # print(">>>>>> Feature Group Deletion Completed <<<<<<")
        
        create_or_update_feature_groups_in_hopsworks(self.hopsworks_api_key, cleaned_data, version)
        
        
        for name, data in cleaned_data.items():
            print(f"Cleaned data for {name}:")
            print(data)

if __name__ == "__main__":
    try:
        print(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        ingestion_cleaning_process = DataIngestionAndCleaning()
        ingestion_cleaning_process.main()
        print(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        raise e

 