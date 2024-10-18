import configparser
import os
import sys
import traceback
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_directory)
from src.utils.config_utils import *
from src.utils.db_utlis import *
from src.utils.feature_group import *
from src.components.data_merging import DataMerger

STAGE_NAME = "Data Processing"

class DataProcessingPipeline:
    def __init__(self):
        self.config = read_config()
        self.hopsworks_api_key = self.config.get('HOPSWORK', 'feature_group_api')
        self.merger = DataMerger(hopsworks_api_key=self.hopsworks_api_key)
        self.model_dir = self.config.get('MODEL', 'model_dir')

    def main(self):
        feature_group_names = [
            'city_weather', 'routes_weather', 'traffic_table',
            'truck_schedule_table', 'trucks_table', 'drivers_table', 'routes_table'
        ]
        # Fetching all feature groups
        feature_groups = fetch_feature_groups(self.hopsworks_api_key, feature_group_names, version=1)
        print("All available feature groups processed.")
        
        # Clean datetime columns for specific dataframes
        feature_groups = self.merger.clean_datetime_columns(feature_groups)

        # Remove the 'eventtime' column from each DataFrame
        feature_groups = self.merger.remove_unnecessary_columns(feature_groups)
        
        # Remove duplicates from each DataFrame based on predefined keys
        feature_groups = self.merger.remove_duplicates(feature_groups)
        
        for name, df in feature_groups.items():
            print(f"Shape of '{name}' after removing duplicates: {df.shape}")
            
        # Create a copy of the truck schedule table and adjust time and date range
        if 'truck_schedule_table' in feature_groups:
            print("Creating and processing a copy of truck schedules.")
            truck_schedule_copy = self.merger.copy_dataframe(feature_groups['truck_schedule_table'])
            truck_schedule_copy = self.merger.adjust_and_explode_truck_schedule(truck_schedule_copy)
            
        # merge with route weather
        if 'routes_weather' in feature_groups:
            merged_truck_weather = self.merger.merge_dataframes(
                truck_schedule_copy,
                feature_groups['routes_weather'],
                keys=['route_id', 'date'],
                how='left'
            )
            merged_truck_weather = self.merger.remove_duplicates(merged_truck_weather)
            merged_truck_weather = self.merger.remove_rows_with_nulls(merged_truck_weather)
            print("merge truck schedule with route weather", merged_truck_weather.shape)
            
            # Merge truck schedule with grouped data 
            grouped_data = self.merger.group_and_aggregate_data(merged_truck_weather)
            print("Grouped and aggregated data shape:", grouped_data.shape)
            schedule_weather_merge = self.merger.merge_dataframes(
                feature_groups['truck_schedule_table'],
                grouped_data,
                keys=['truck_id', 'route_id'],
                how='left'
            )
            schedule_weather_merge = self.merger.remove_duplicates(schedule_weather_merge)
            schedule_weather_merge = self.merger.remove_rows_with_nulls(schedule_weather_merge)
            print("merge truck schedule with grouped data", schedule_weather_merge.shape)
            
            # Merge truck schedule with route table
            nearest_hour_schedule_df = self.merger.copy_dataframe(feature_groups['truck_schedule_table'])
            nearest_hour_schedule_df = self.merger.round_datetime_columns(
                nearest_hour_schedule_df,
                ['estimated_arrival', 'departure_date']
            )
            if 'routes_table' in feature_groups:
                nearest_hour_schedule_route_df = self.merger.merge_dataframes(
                nearest_hour_schedule_df,
                feature_groups['routes_table'],
                keys='route_id',  # Correct usage of the 'keys' parameter
                how='left'
            )
            
            nearest_hour_schedule_route_df = self.merger.remove_duplicates(nearest_hour_schedule_route_df)
            nearest_hour_schedule_route_df = self.merger.remove_rows_with_nulls(nearest_hour_schedule_route_df)
            print("Merged and cleaned nearest hour schedule with routes. Data shape:", nearest_hour_schedule_route_df.shape)

            # merge origin weather and destination weather
            if 'city_weather' in feature_groups:
                # Handle origin and destination weather data
                origin_weather_data = self.merger.copy_dataframe(feature_groups['city_weather'])
                destination_weather_data = self.merger.copy_dataframe(feature_groups['city_weather'])
                
                origin_weather_merger = self.merger.merge_dataframes(
                nearest_hour_schedule_route_df,
                origin_weather_data,
                left_key=['origin_id', 'departure_date_nearest_hour'],
                right_key=['city_id', 'datetime'],
                how='left'
                )
                origin_weather_merger = self.merger.remove_duplicates(origin_weather_merger)
                origin_weather_merger = self.merger.remove_rows_with_nulls(origin_weather_merger)
                print('Merge truck schedule with origin weather shape ',origin_weather_merger.shape)
                
                destination_weather_merge = self.merger.merge_dataframes(
                origin_weather_merger,  
                destination_weather_data,  
                left_key=['destination_id', 'estimated_arrival_nearest_hour'],
                right_key=['city_id', 'datetime'],
                suffixes=('_origin', '_destination')
                )
                destination_weather_merge = self.merger.remove_duplicates(destination_weather_merge)
                destination_weather_merge = self.merger.remove_rows_with_nulls(destination_weather_merge)
                print('Merge truck schedule with destination weather shape ',destination_weather_merge.shape)
                
                # Merge truck schedule and traffic
                schedule_copy = self.merger.copy_dataframe(feature_groups['truck_schedule_table'])
                columns_to_round = ['departure_date', 'estimated_arrival']
                schedule_copy = self.merger.round_datetime_columns(schedule_copy, columns_to_round)
                hourly_exploded_schedule_df = self.merger.explode_hourly_schedule(
                nearest_hour_schedule_df, 'departure_date', 'estimated_arrival'
                )
                
                traffic_df = feature_groups['traffic_table']
                scheduled_traffic = self.merger.merge_dataframes(
                    hourly_exploded_schedule_df, 
                    traffic_df,
                    left_key=['route_id', 'custom_date'], 
                    right_key=['route_id', 'datetime'],
                    how='left'
                )
                scheduled_traffic = self.merger.remove_duplicates(scheduled_traffic)
                scheduled_traffic = self.merger.remove_rows_with_nulls(scheduled_traffic)
                print('Merge truck schedule with traffic shape ',scheduled_traffic.shape)
                
                # Aggregated traffic data
                scheduled_route_traffic = self.merger.aggregate_traffic_data(scheduled_traffic)
                print("Aggregated traffic data:", scheduled_route_traffic.shape)
                
                # Merge destination_weather_merge and scheduled_route_traffic
                origin_destination_weather_traffic_merge = self.merger.merge_dataframes(
                destination_weather_merge,
                scheduled_route_traffic,
                keys=['truck_id', 'route_id'],
                how='left'
                )
                origin_destination_weather_traffic_merge = self.merger.remove_duplicates(origin_destination_weather_traffic_merge)
                origin_destination_weather_traffic_merge = self.merger.remove_rows_with_nulls(origin_destination_weather_traffic_merge)
                print('Merge truck schedule,weather  with traffic shape ',origin_destination_weather_traffic_merge.shape)
                    
                # Merge the schedule weather with origin_destination_weather_traffic_merge
                merged_data_weather_traffic = self.merger.merge_dataframes(
                schedule_weather_merge,
                origin_destination_weather_traffic_merge,
                keys=['truck_id', 'route_id', 'departure_date', 'estimated_arrival', 'delay'],
                how='left'
                )
                merged_data_weather_traffic = self.merger.remove_duplicates(merged_data_weather_traffic)
                merged_data_weather_traffic = self.merger.remove_rows_with_nulls(merged_data_weather_traffic)
                print('Merge schedule_weather_merge and origin_destination_weather_traffic_merge ',merged_data_weather_traffic.shape)
                
                
                # Merge weather traffic with truck
                trucks_df = feature_groups['trucks_table']
                merged_data_weather_traffic_trucks = self.merger.merge_dataframes(
                merged_data_weather_traffic,
                trucks_df,
                keys='truck_id',
                how='left'
                )
                merged_data_weather_traffic_trucks = self.merger.remove_duplicates(merged_data_weather_traffic_trucks)
                merged_data_weather_traffic_trucks = self.merger.remove_rows_with_nulls(merged_data_weather_traffic_trucks)
                print('Merge merged_data_weather_traffic_trucks and truck ',merged_data_weather_traffic_trucks.shape)

                # Merge with driver data
                drivers_df = feature_groups['drivers_table']
                final_merge_dataset = self.merger.merge_dataframes(
                merged_data_weather_traffic_trucks,
                drivers_df,
                left_key='truck_id',
                right_key='vehicle_no',
                how='left'
                )
                
                final_merge_dataset['is_midnight'] = final_merge_dataset.apply(
                lambda row: self.merger.has_midnight(row['departure_date'], row['estimated_arrival']),
                axis=1
                )
                final_merge_dataset = self.merger.remove_rows_with_nulls(final_merge_dataset)
             
                
                columns_to_remove = ['date','city_id_origin', 'city_id_destination', 
                                    'datetime_origin','datetime_destination']
                final_merge_dataset = self.merger.drop_columns(final_merge_dataset, columns_to_remove)
                
                final_merge_dataset = self.merger.remove_duplicates(final_merge_dataset)
            
                print('The final_merge_dataset shape ',final_merge_dataset.shape)
                print("The Final merge data columns are ",final_merge_dataset.columns)
                
                
                final_merge_dataset = self.merger.add_sequential_id(final_merge_dataset)
                
                # feature_group_names = [
                #     'final_preprocessed_data'
                # ]
                # delete_feature_groups(self.hopsworks_api_key, feature_group_names, version=1)
               
                create_or_update_feature_groups_in_hopsworks(
                    hopsworks_api_key=self.hopsworks_api_key,
                    cleaned_data={'final_dataset': final_merge_dataset},
                    version=1
                )
                print("Feature group creation/update completed.")
            

if __name__ == "__main__":
    try:
        print(f">>>>>> Stage {STAGE_NAME} starting <<<<<<")
        processing_pipeline = DataProcessingPipeline()
        processing_pipeline.main()
        print(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
