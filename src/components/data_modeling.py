import pandas as pd
import numpy as np
import hopsworks
import traceback
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import mlflow
import logging
import os
import pickle
import logging
import mlflow.sklearn


# Configure logging at the top of your script
logging.basicConfig(level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Example use at the start of your script or main execution block
logging.info("Script started successfully.")


class DataModeling:

    
    # Define continuous and categorical columns and target as class variables
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
    target = 'delay'
    
    def __init__(self, hopsworks_api_key,model_dir):
        self.hopsworks_api_key = hopsworks_api_key
        self.model_dir = model_dir
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.scaler = StandardScaler()
        logging.info("DataModeling instance created with Hopsworks API key.")

    def check_nulls(self, df):
        null_stats = df.isnull().sum()
        logging.info("Checking for null values in the DataFrame.")
        if null_stats.any():
            logging.warning(f"Null values found: {null_stats[null_stats > 0]}")
        else:
            logging.info("No null values found in the DataFrame.")

    def convert_datetime_columns(self, df, columns):
        logging.info(f"Converting columns {columns} to datetime format.")
        for column in columns:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column], errors='coerce').dt.tz_localize(None)
            else:
                logging.warning(f"Column {column} not found in DataFrame.")
        return df
    
    def prepare_datasets(self, df, date_column, split_dates, cts_cols, cat_cols, target):
      
        # Convert date column to datetime if not already
        df[date_column] = pd.to_datetime(df[date_column])

        # Splitting the data into train, validation, and test sets
        train_df = df[df[date_column] <= split_dates[0]]
        validation_df = df[(df[date_column] > split_dates[0]) & (df[date_column] <= split_dates[1])]
        test_df = df[df[date_column] > split_dates[1]]

        # Resetting the index for alignment
        train_df.reset_index(drop=True, inplace=True)
        validation_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        # Preparing feature and target sets
        X_train = train_df[cts_cols + cat_cols]
        y_train = train_df[target]
        X_valid = validation_df[cts_cols + cat_cols]
        y_valid = validation_df[target]
        X_test = test_df[cts_cols + cat_cols]
        y_test = test_df[target]

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_valid': X_valid, 'y_valid': y_valid,
            'X_test': X_test, 'y_test': y_test
        }
    
    def encode_features(self, X_train, X_valid, X_test, encode_columns):
        logging.info("Using initialized OneHotEncoder.")
        
        # Fit the encoder on the training data
        logging.debug(f"Fitting encoder on training data columns: {encode_columns}")
        self.encoder.fit(X_train[encode_columns])

        # Get the new feature names from the encoder
        encoded_features = list(self.encoder.get_feature_names_out(encode_columns))

        # Transform the training, validation, and test datasets
        logging.debug("Transforming the training dataset.")
        X_train_encoded = self.encoder.transform(X_train[encode_columns])
        X_train = pd.concat([X_train.drop(encode_columns, axis=1),
                            pd.DataFrame(X_train_encoded, columns=encoded_features, index=X_train.index)], axis=1)

        logging.debug("Transforming the validation dataset.")
        X_valid_encoded = self.encoder.transform(X_valid[encode_columns])
        X_valid = pd.concat([X_valid.drop(encode_columns, axis=1),
                            pd.DataFrame(X_valid_encoded, columns=encoded_features, index=X_valid.index)], axis=1)

        logging.debug("Transforming the test dataset.")
        X_test_encoded = self.encoder.transform(X_test[encode_columns])
        X_test = pd.concat([X_test.drop(encode_columns, axis=1),
                            pd.DataFrame(X_test_encoded, columns=encoded_features, index=X_test.index)], axis=1)

        return X_train, X_valid, X_test
        
    def scale_features(self, X_train, X_valid, X_test, continuous_cols):
        logging.info("Using initialized StandardScaler.")
        
        # Fit the scaler on the training data
        logging.debug(f"Fitting scaler on training data columns: {continuous_cols}")
        self.scaler.fit(X_train[continuous_cols])

        # Transform the validation and test datasets
        logging.debug("Transforming the validation dataset.")
        X_valid[continuous_cols] = self.scaler.transform(X_valid[continuous_cols])

        logging.debug("Transforming the test dataset.")
        X_test[continuous_cols] = self.scaler.transform(X_test[continuous_cols])

        return X_train, X_valid, X_test

    def get_model_info(self):
        models_info = {
              'Random_Forest': (RandomForestClassifier(), {
            'n_estimators': [100, 200],  # Increase number of trees
            'max_depth': [5, 10],  # Limit the depth of trees
            'min_samples_split': [5, 10],  # Increase minimum samples required to split
            'min_samples_leaf': [3, 5],  # Increase minimum samples required at a leaf
            'max_features': ['sqrt', 'log2'],  # Subset of features to consider at each split
            'class_weight': [{0: 1, 1: 2}, {0: 1, 1: 3}]  # Assign higher weight to class 1
        }),
            'Logistic_Regression': (LogisticRegression(max_iter=300), {  # Reduced iterations for faster convergence
                'C': [0.1, 1],  # Simplified range for regularization
                'solver': ['liblinear'],  # Faster solver for small datasets
                'class_weight': ['balanced']  # Handle class imbalance
            }),
            'XGBoost': (XGBClassifier(), {
            'n_estimators': [50, 100],  # Reduce the number of trees
            'learning_rate': [0.01, 0.05],  # Lower learning rate for smoother convergence
            'max_depth': [3, 5],  # Limit tree depth
            'subsample': [0.5, 0.8],  # Reduce subsample ratio
            'colsample_bytree': [0.8],  # Keep feature sampling
            'reg_lambda': [1, 5],  # Increase L2 regularization
            'reg_alpha': [1, 5],  # Increase L1 regularization
            'min_child_weight': [3, 5, 7],  # Add weight constraint on children
            'gamma': [0.1, 0.5, 1]
            })
        }
        return models_info



    def run_model_training(self, model, param_grid, model_name, X_train, y_train, X_valid, y_valid, X_test, y_test):
        try:
            # Initialize GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
            
            # Fitting the model with training data
            logging.info(f"Training {model_name} with GridSearchCV...")
            grid_search.fit(X_train, y_train)

            # Log the best parameters found by GridSearchCV
            best_params = grid_search.best_params_
            logging.info(f"Best parameters for {model_name}: {best_params}")

            # Predict on the training set and print the classification report
            train_predictions = grid_search.predict(X_train)
            train_report = classification_report(y_train, train_predictions)
            print(f"Training Classification Report for {model_name}:\n{train_report}")
            # Calculate accuracy for training
            train_accuracy = accuracy_score(y_train, train_predictions)
            logging.info(f"Training Accuracy for {model_name}: {train_accuracy}")

            # Predict on the validation set and print the classification report
            valid_predictions = grid_search.predict(X_valid)
            valid_report = classification_report(y_valid, valid_predictions)
            print(f"Validation Classification Report for {model_name}:\n{valid_report}")
            # Calculate accuracy for validation
            valid_accuracy = accuracy_score(y_valid, valid_predictions)
            logging.info(f"Validation Accuracy for {model_name}: {valid_accuracy}")

            # Predict on the test set and print the classification report
            test_predictions = grid_search.predict(X_test)
            test_report = classification_report(y_test, test_predictions)
            print(f"Test Classification Report for {model_name}:\n{test_report}")
            # Calculate accuracy for test
            test_accuracy = accuracy_score(y_test, test_predictions)
            logging.info(f"Test Accuracy for {model_name}: {test_accuracy}")

            # Return the grid search object along with accuracies for further evaluation
            return train_accuracy, valid_accuracy, test_accuracy, grid_search

        except Exception as e:
            logging.error(f"An error occurred while training {model_name}: {e}", exc_info=True)
            raise  # Re-raise the exception for debugging


    def find_and_save_best_model(self, models_info, X_train, y_train, X_valid, y_valid, X_test, y_test):
        best_model_name = None
        best_accuracy = 0  # Initialize with a logical zero for comparison
        best_model_details = {}

        # Iterate through each model and parameter grid
        for model_name, (model, param_grid) in models_info.items():
            logging.info(f"Training model: {model_name}")
            train_accuracy, valid_accuracy, test_accuracy, grid_search = self.run_model_training(
                model, param_grid, model_name, X_train, y_train, X_valid, y_valid, X_test, y_test
            )
            
            # Track the best model based on validation accuracy
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_model_name = model_name
                best_model_details = {
                    'model': grid_search.best_estimator_,
                    'encoder': self.encoder,
                    'scaler': self.scaler,
                    'train_accuracy': train_accuracy,
                    'valid_accuracy': valid_accuracy,
                    'test_accuracy': test_accuracy,
                    'grid_search': grid_search
                }

        if best_model_name:
            logging.info(f"Best model identified: {best_model_name} with validation accuracy: {best_accuracy}")
            self.save_and_log_model(best_model_name, best_model_details, X_train)

        else:
            logging.warning("No model was selected as the best model.")

    def save_and_log_model(self, best_model_name, model_details, X_train):
        # Define local file paths for saving
        model_dir = self.model_dir  # Ensure this is defined or passed appropriately
        model_file = os.path.join(model_dir, f"{best_model_name}_model.pkl")
        encoder_file = os.path.join(model_dir, f"{best_model_name}_encoder.pkl")
        scaler_file = os.path.join(model_dir, f"{best_model_name}_scaler.pkl")

        # Save the model, encoder, and scaler locally using pickle
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(model_details['model'], f)
            with open(encoder_file, 'wb') as f:
                pickle.dump(model_details['encoder'], f)
            with open(scaler_file, 'wb') as f:
                pickle.dump(model_details['scaler'], f)

            # Log details with MLflow
            self.log_model_to_mlflow(model_details, best_model_name, X_train, model_file, encoder_file, scaler_file)

        except Exception as e:
            logging.error(f"An error occurred while saving and logging model artifacts: {e}", exc_info=True)
            raise

    def log_model_to_mlflow(self, model_details, best_model_name, X_train, model_file, encoder_file, scaler_file):
        with mlflow.start_run(run_name=f"Best Model - {best_model_name}"):
            # Log the best model with signature
            signature = mlflow.models.infer_signature(X_train, model_details['grid_search'].predict(X_train))
            mlflow.sklearn.log_model(
                model_details['model'],
                artifact_path=best_model_name,
                signature=signature
            )
            mlflow.log_params(model_details['grid_search'].best_params_)
            mlflow.log_metrics({
                'train_accuracy': model_details['train_accuracy'],
                'validation_accuracy': model_details['valid_accuracy'],
                'test_accuracy': model_details['test_accuracy']
            })
            mlflow.log_artifact(encoder_file, artifact_path="encoder")
            mlflow.log_artifact(scaler_file, artifact_path="scaler")