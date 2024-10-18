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
import shutil
import pickle

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.info("Script started successfully.")


class DataModeling:
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
    
    def __init__(self, hopsworks_api_key):
        self.hopsworks_api_key = hopsworks_api_key
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
      
        df[date_column] = pd.to_datetime(df[date_column])
        train_df = df[df[date_column] <= split_dates[0]]
        validation_df = df[(df[date_column] > split_dates[0]) & (df[date_column] <= split_dates[1])]
        test_df = df[df[date_column] > split_dates[1]]

        train_df.reset_index(drop=True, inplace=True)
        validation_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
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
        
        logging.info("Initializing OneHotEncoder.")
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        logging.debug(f"Fitting encoder on training data columns: {encode_columns}")
        encoder.fit(X_train[encode_columns])
        encoded_features = list(encoder.get_feature_names_out(encode_columns))

        # Transform the training, validation, and test datasets
        logging.debug("Transforming the training dataset.")
        X_train_encoded = encoder.transform(X_train[encode_columns])
        X_train = pd.concat([X_train.drop(encode_columns, axis=1),
                             pd.DataFrame(X_train_encoded, columns=encoded_features, index=X_train.index)], axis=1)

        logging.debug("Transforming the validation dataset.")
        X_valid_encoded = encoder.transform(X_valid[encode_columns])
        X_valid = pd.concat([X_valid.drop(encode_columns, axis=1),
                             pd.DataFrame(X_valid_encoded, columns=encoded_features, index=X_valid.index)], axis=1)

        logging.debug("Transforming the test dataset.")
        X_test_encoded = encoder.transform(X_test[encode_columns])
        X_test = pd.concat([X_test.drop(encode_columns, axis=1),
                            pd.DataFrame(X_test_encoded, columns=encoded_features, index=X_test.index)], axis=1)

        return X_train, X_valid, X_test
    
    def scale_features(self, X_train, X_valid, X_test, continuous_cols):
        
        logging.info("Initializing StandardScaler.")
        scaler = StandardScaler()
        logging.debug(f"Fitting scaler on training data columns: {continuous_cols}")
        X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])

        logging.debug("Transforming the validation dataset.")
        X_valid[continuous_cols] = scaler.transform(X_valid[continuous_cols])

        logging.debug("Transforming the test dataset.")
        X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

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


    import pandas as pd

    def run_model_training(self, model, param_grid, model_name, X_train, y_train, X_valid, y_valid, X_test, y_test):
        try:
            # Start a new MLflow run for each model training session
            with mlflow.start_run(run_name=model_name):
                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
                
                # Fitting the model with training data
                logging.info(f"Training {model_name} with GridSearchCV...")
                grid_search.fit(X_train, y_train)

                # Log the best parameters found by GridSearchCV
                best_params = grid_search.best_params_
                logging.info(f"Best parameters for {model_name}: {best_params}")
                mlflow.log_params(best_params)  # Log parameters to MLflow

                # Predict on the training set and format the classification report
                train_predictions = grid_search.predict(X_train)
                train_report = classification_report(y_train, train_predictions, output_dict=True)
                train_report_df = pd.DataFrame(train_report).transpose()
                print(f"\nTraining Classification Report for {model_name}:\n{train_report_df[['precision', 'recall', 'f1-score', 'support']]}\n")
                train_accuracy = train_report['accuracy']

                # Predict on the validation set and format the classification report
                valid_predictions = grid_search.predict(X_valid)
                valid_report = classification_report(y_valid, valid_predictions, output_dict=True)
                valid_report_df = pd.DataFrame(valid_report).transpose()
                print(f"\nValidation Classification Report for {model_name}:\n{valid_report_df[['precision', 'recall', 'f1-score', 'support']]}\n")
                valid_accuracy = valid_report['accuracy']

                # Predict on the test set and format the classification report
                test_predictions = grid_search.predict(X_test)
                test_report = classification_report(y_test, test_predictions, output_dict=True)
                test_report_df = pd.DataFrame(test_report).transpose()
                print(f"\nTest Classification Report for {model_name}:\n{test_report_df[['precision', 'recall', 'f1-score', 'support']]}\n")
                test_accuracy = test_report['accuracy']

                # Log metrics for each dataset using the existing function
                self.log_classification_metrics(train_report, "train")
                self.log_classification_metrics(valid_report, "validation")
                self.log_classification_metrics(test_report, "test")

                # Log the trained model with its signature
                signature = mlflow.models.infer_signature(X_train, train_predictions)
                mlflow.sklearn.log_model(grid_search.best_estimator_, artifact_path=model_name, signature=signature)
                logging.info(f"Model for {model_name} logged to MLflow.")
                
                # Register the Random Forest model if the model_name is "Random Forest"
                if model_name == "Random_Forest":
                    metrics = {
                        "test_accuracy": test_accuracy,
                        "test_precision_class_0": test_report['0']['precision'],
                        "test_recall_class_0": test_report['0']['recall'],
                        "test_f1_score_class_0": test_report['0']['f1-score'],
                        "test_support_class_0": test_report['0']['support'],
                        "test_precision_class_1": test_report['1']['precision'],
                        "test_recall_class_1": test_report['1']['recall'],
                        "test_f1_score_class_1": test_report['1']['f1-score'],
                        "test_support_class_1": test_report['1']['support'],
                        "test_f1_score_weighted": test_report['weighted avg']['f1-score'],
                        "test_precision_weighted": test_report['weighted avg']['precision'],
                        "test_recall_weighted": test_report['weighted avg']['recall']
                    }
                    self.register_random_forest_in_hopsworks("Random_Forest", metrics, grid_search.best_estimator_)
            
        except Exception as e:
            logging.error(f"An error occurred while training {model_name}: {e}", exc_info=True)
            raise 
        finally:
            mlflow.end_run()

        return train_accuracy, valid_accuracy, test_accuracy

    def log_classification_metrics(self, report, dataset_type):
        
        for class_label, metrics in report.items():
            if isinstance(metrics, dict):  
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{dataset_type}_{class_label}_{metric_name}", value)


    def register_random_forest_in_hopsworks(self, model_name, metrics, model):
        try:
            project = hopsworks.login(api_key_value=self.hopsworks_api_key)
            model_registry = project.get_model_registry()

            model_dir = "/Users/jeevanapathipati/Documents/TruckDelay/models/"
            model_path = os.path.join(model_dir, f"{model_name}.pkl")
            if os.path.exists(model_path):
                os.remove(model_path)
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logging.info(f"Random Forest model saved as a pickle file at {model_path}.")

            model_version = model_registry.python.create_model(
                name=model_name,
                version=1, 
                metrics=metrics,  
                description=f"Random Forest model with metrics: {metrics}"
            )
            
            model_version.save(model_path) 
            logging.info(f"Random Forest model registered successfully in Hopsworks with version {model_version.version}.")

        except Exception as e:
            logging.error(f"An error occurred while registering Random Forest: {e}", exc_info=True)
