import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import hopsworks
import traceback
import warnings
import time
import random
import hsfs

class DataCleaner:
    def __init__(self,hopsworks_api_key):
        self.knn_imputer = KNNImputer()
        self.hopsworks_api_key = hopsworks_api_key

    def drop_columns(self, df, columns):
        existing_columns = [col for col in columns if col in df.columns]
        if existing_columns:
            df = df.drop(columns=existing_columns)
            print(f"Dropped columns: {existing_columns}")
        else:
            print(f"No columns to drop from: {columns}")
        return df

    def remove_duplicates(self, df):
        return df.drop_duplicates(keep='first')

    def filling_missing_values(self, df):
        numeric_cols = df.select_dtypes(include=['number']).columns
        non_numeric_cols = df.select_dtypes(include=['object']).columns

        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = self.knn_imputer.fit_transform(df[[col]])

        for col in non_numeric_cols:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode()[0]
                df[col] = df[col].fillna(mode_value)
        
        return df

   
    def remove_outliers(self, df, table_name):
        exclusions = {
            'city_weather': ['precip', 'visibility'],  
            'routes_weather': ['precip', 'visibility'],  
            'drivers_table': ['vehicle_no'],  
            'trucks_table': ['truck_id', 'load_capacity_pounds', 'mileage_mpg', 'fuel_type'],  
            'truck_schedule_table': []  
        }

        excluded_columns = exclusions.get(table_name, [])
        print(f"Excluding columns for {table_name}: {excluded_columns}")
        if table_name in ['trucks_table']:  
            target_columns = ['truck_age']
        elif table_name == 'truck_schedule_table':
            return df  
        else:
            target_columns = [col for col in df.columns if col not in excluded_columns and df[col].dtype.kind in 'biufc']

        # Remove outliers from the target columns
        for column in target_columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            print(f"Outliers removed from {column} in {table_name}.")

        return df


    def exp(self, df):
        mean_experience = df['experience'][df['experience'] >= 0].mean()
        df['experience'] = df['experience'].apply(lambda x: mean_experience if x < 0 else x)
        return df

    def convert_estimated_arrival(self, df, column_name):
        df[column_name] = pd.to_datetime(df[column_name]).dt.floor('s')
        return df

    def add_index_and_eventtime_columns(self, df):
        df['index'] = range(len(df))
        df['eventtime'] = pd.to_datetime(pd.Timestamp.now().floor('s'))
        return df

    def combine_date_hour(self, df, date_col, hour_col, new_col_name='datetime', default_sec='00'):
        df[hour_col] = df[hour_col].astype(str).str.zfill(4)
        df[new_col_name] = pd.to_datetime(df[date_col].astype(str) + ' ' +
                                          df[hour_col].str.slice(0, 2) + ':' +
                                          df[hour_col].str.slice(2, 4) + ':' +
                                          default_sec).dt.floor('s')
        return df
    
    def convert_string_to_datetime(self, df, table_name):
        warnings.simplefilter(action='ignore', category=UserWarning)
        date_column_patterns = ['date', 'time', 'year', 'month', 'day']

        def is_date_column(col_name, col_data):
            if any(pattern in col_name.lower() for pattern in date_column_patterns):
                try:
                    pd.to_datetime(col_data, errors='raise')
                    return True
                except:
                    return False
            return False

        for col_name in df.columns:
            if df[col_name].dtype == 'object' and is_date_column(col_name, df[col_name]):
                try:
                    df[col_name] = pd.to_datetime(df[col_name], errors='coerce').dt.floor('s')
                except Exception as e:
                    print(f"Failed to convert {col_name} in table '{table_name}': {e}")

        return df
    

   