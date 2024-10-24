import configparser
import os
import sys
import mlflow
import traceback
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_directory)
from src.utils.config_utils import *
from src.utils.feature_group import *
from src.components.data_modeling import DataModeling
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG,  # Capture all levels of logs
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

STAGE_NAME = "Data Modeling"

class DataModelingPipeline:
    def __init__(self):
        logging.debug("Reading configuration settings.")
        self.config = read_config()
        self.hopsworks_api_key = self.config.get('HOPSWORK', 'feature_group_api')
        self.model_dir = self.config.get('MODEL','model_dir')
        self.modeling = DataModeling(hopsworks_api_key=self.hopsworks_api_key,model_dir=self.model_dir)
        logging.info("DataModelingPipeline initialized with Hopsworks API key.")

    def main(self):
        logging.info("Starting the main function of the Data Modeling pipeline.")
        
        # Set the MLflow experiment
        mlflow.set_experiment("truck_delay_experiment")
        logging.info("MLflow experiment set to 'truck_delay_experiment'.")
        
        feature_group_names = ['final_dataset']
        logging.debug("Fetching feature groups from Hopsworks.")
        feature_groups = fetch_feature_groups(self.hopsworks_api_key, feature_group_names, version=1)
        logging.info("Feature groups fetched successfully.")

        truck_delay_df = feature_groups['final_dataset']
        logging.debug("Checking for null values in the dataset.")
        self.modeling.check_nulls(truck_delay_df)

        logging.info("Converting datetime columns.")
        truck_delay_df = self.modeling.convert_datetime_columns(truck_delay_df, ['estimated_arrival'])

        # Splitting dates for train, validation, and test sets
        split_dates = [pd.to_datetime('2019-01-30'), pd.to_datetime('2019-02-07')]
    
        # Prepare datasets
        datasets = self.modeling.prepare_datasets(truck_delay_df, 'estimated_arrival', split_dates,
                                                DataModeling.cts_cols, DataModeling.cat_cols, 'delay')
    
        # Extracting datasets
        X_train, y_train = datasets['X_train'], datasets['y_train']
        X_valid, y_valid = datasets['X_valid'], datasets['y_valid']
        X_test, y_test = datasets['X_test'], datasets['y_test']

        # Encode categorical features
        encode_columns = ['route_description', 'description_origin', 'description_destination', 'fuel_type', 'gender', 'driving_style']
        X_train, X_valid, X_test = self.modeling.encode_features(X_train, X_valid, X_test, encode_columns)

        # Scale continuous features
        continuous_columns = DataModeling.cts_cols
        X_train, X_valid, X_test = self.modeling.scale_features(X_train, X_valid, X_test, continuous_columns)

        # Train and evaluate models
        logging.info("Training and evaluating models.")
        models_info = self.modeling.get_model_info()  # Retrieve model configurations
        
        # Find and save the best model
        self.modeling.find_and_save_best_model(models_info, X_train, datasets['y_train'], X_valid, datasets['y_valid'], X_test, datasets['y_test'])

if __name__ == "__main__":
    logging.info(f">>>>>> Stage {STAGE_NAME} starting <<<<<<")
    try:
        pipeline = DataModelingPipeline()
        pipeline.main()
        logging.info(f">>>>>> Stage {STAGE_NAME} completed successfully <<<<<<")
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

