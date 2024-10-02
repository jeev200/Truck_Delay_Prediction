import pandas as pd
import hopsworks
import numpy as np

class DataMerger:
    def __init__(self, hopsworks_api_key):
        self.hopsworks_api_key = hopsworks_api_key
        
        
    def clean_datetime_columns(self, feature_groups):
        datetime_cleaning_operations = {
            'city_weather': 'datetime',
            'traffic_table': 'datetime',
            'routes_weather': 'date',
            'truck_schedule_table': ['estimated_arrival', 'departure_date']
        }
        for name, columns in datetime_cleaning_operations.items():
            if name in feature_groups:
                if isinstance(columns, list):
                    for column in columns:
                        feature_groups[name] = self.clean_single_datetime_column(feature_groups[name], column)
                else:
                    feature_groups[name] = self.clean_single_datetime_column(feature_groups[name], columns)
        return feature_groups

    def clean_single_datetime_column(self, df, date_column):
        df[date_column] = df[date_column].astype(str).str.split('+').str[0].str.split('.').str[0]
        df[date_column] = pd.to_datetime(df[date_column], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        return df
    
    
    def remove_unnecessary_columns(self, df_dict):
        # Removes the 'eventtime' and 'index' columns from each DataFrame, if they exist
        for name, df in df_dict.items():
            columns_to_remove = [col for col in ['eventtime', 'index'] if col in df.columns]
            if columns_to_remove:
                df.drop(columns=columns_to_remove, inplace=True)
                print(f"Columns {', '.join(columns_to_remove)} removed from {name}")
        return df_dict
    
    def remove_duplicates(self, df_dict):
        # Mapping of DataFrame keys to specific columns to check for duplicates
        subsets = {
            'city_weather': ['city_id', 'datetime'],
            'routes_weather': ['route_id', 'date'],
            'trucks_table': ['truck_id'],
            'drivers_table': ['driver_id'],
            'routes_table': ['route_id', 'destination_id', 'origin_id'],
            'truck_schedule_table': ['truck_id', 'route_id', 'departure_date']
        }

        # Iterate over each DataFrame in the dictionary and apply drop_duplicates
        for key, df in df_dict.items():
            if key in subsets:
                df_dict[key] = df.drop_duplicates(subset=subsets[key])

        return df_dict
    
    def copy_dataframe(self, df):
        return df.copy()

    def adjust_and_explode_truck_schedule(self, df):
        df['estimated_arrival'] = pd.to_datetime(df['estimated_arrival']).dt.ceil('6H')
        df['departure_date'] = pd.to_datetime(df['departure_date']).dt.floor('6H')
        df['date'] = [pd.date_range(start, end, freq='6H').tolist() for start, end in zip(df['departure_date'], df['estimated_arrival'])]
        return df.explode('date')
    
    def custom_mode(self, x):
        modes = x.mode()
        if not modes.empty:
            return modes.iloc[0]
        else:
            return np.nan
        
    def group_and_aggregate_data(self, df):
        grouped_df = df.groupby(['truck_id', 'route_id'], as_index=False).agg(
            route_avg_temp=('temp', 'mean'),
            route_avg_wind_speed=('wind_speed', 'mean'),
            route_avg_precip=('precip', 'mean'),
            route_avg_humidity=('humidity', 'mean'),
            route_avg_visibility=('visibility', 'mean'),
            route_avg_pressure=('pressure', 'mean'),
            route_description=('description', self.custom_mode)
        )
        return grouped_df
    

    def round_datetime_columns(self, df, columns, freq="H"):
        for col in columns:
            if col in df.columns:
                df[col + '_nearest_hour'] = df[col].dt.round(freq)
        return df
    

    def merge_dataframes(self, df1, df2, left_key=None, right_key=None, keys=None, how='left', suffixes=('_left', '_right')):
    
        try:
            if left_key and right_key:
                merged_df = pd.merge(df1, df2, left_on=left_key, right_on=right_key, how=how, suffixes=suffixes)
            elif keys:
                merged_df = pd.merge(df1, df2, on=keys, how=how, suffixes=suffixes)
            else:
                raise ValueError("Either left_key and right_key or keys must be provided.")
            return merged_df
        except KeyError as e:
            print(f"KeyError: {e} - Check that the merge keys exist in both DataFrames.")
            return None
        except ValueError as e:
            print(f"ValueError: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def remove_rows_with_nulls(self, df):
        
        original_shape = df.shape
        df_cleaned = df.dropna()
        new_shape = df_cleaned.shape
        print(f"Removed rows with nulls: {original_shape[0] - new_shape[0]} rows dropped. New shape: {new_shape}")
        return df_cleaned
    
    def drop_columns(self, df, columns_to_drop):
        return df.drop(columns=columns_to_drop, errors='ignore')
    
    def explode_hourly_schedule(self, df, start_col, end_col, new_col_name='custom_date'):
        df[new_col_name] = [
            pd.date_range(start, end, freq='H').tolist()
            for start, end in zip(df[start_col], df[end_col])
        ]
        return df.explode(new_col_name, ignore_index=True)
    
    def custom_agg(self, values):
        return 1 if any(values == 1) else 0

    def aggregate_traffic_data(self, df):
        return df.groupby(['truck_id', 'route_id'], as_index=False).agg(
            avg_no_of_vehicles=('no_of_vehicles', 'mean'),
            accident=('accident', self.custom_agg)  # Using the custom aggregation function
        )
        
    def has_midnight(self, start, end):
        return int(start.date() != end.date())
    
    def add_sequential_id(self, df, id_column_name='id'):
        
        df[id_column_name] = range(1, len(df) + 1)
        return df


                
            
            
            
            
            
            
            
            
   