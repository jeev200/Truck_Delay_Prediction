**1 Data Ingestion and Preparation****

  This project is the first part of a three-part series aimed at solving the truck delay prediction problem. In this initial phase, we will utilize PostgreSQL and MYSQL in AWS Redshift to store the data, perform data retrieval, and conduct basic exploratory data analysis (EDA). With Hopsworks feature store, we will build a pipeline that includes data processing feature engineering and prepare the data for model building.

### (a). Data Ingestion Approach
 
* Upload truck delay training data into github repo
* Create MYSQL or Postgress DB Server ( move to AWS RDS in future)
* Create Truck_delays database and create tables for each of csv files
* Develop data_ingestion component to ingest data from github to MYSQL

### (b). Data Exploration 

* Connect to Database
* Fetch Data from truck delay tables into Dataframes
* For each of the DataFrame -
  - Do basic checks of data ( info, describe, etc)
  - Chage Date column to datetime  ( ex : weather_df['date'] = pd.to_datetime(weather_df['date'])
* Data Analysis ( Important : For each of DF analyis - Clearly write your observations and recommandations)
  - Drivers
      * Histogram plots for numeric features
      * Scatter Plots ( Rating vs Average Speed)
      * Box Plot ( Driver Ratings by Gender)
  - Trucks
      * Histogram Plots for numeric features
      * Identify low milege Trucks and plot Trucks with age distribution
  - Routes
    * Histogram plots for numeric values
  - Traffic
    * Histogram plots for numeric values
    * Categorizes hours of the day into time periods.
      Example :
                if 300 <= hour < 600:
                       return 'Early Morning'

### (c). Data cleaning

* For Each of the DataFrame
   - Identify and treat null values
   - Identify and treat outliers

### (d). Create Feature groups in Feature Store


### (e). Fetch Data from feature Store

### (f). Data Preparation



**In Part 2**, we delve deeper into the machine-learning pipeline. Focusing on data retrieval from the feature store, train-validation-test split, one-hot encoding, scaling numerical features, and leveraging MLFlow for model experimentation, we will build our pipeline for model building with logistic regression, random forest, and XGBoost models. Further, we explore hyperparameter tuning , discuss grid and random search, and, ultimately, the deployment of a Streamlit application on AWS.

1. Retrieve Truckdealy feature dataset from Hopsworks

2. Verify Dataset      
                 
3.Check for null values if any and treat them

4.Train-Validation-Test Split

       **** selecting necessary columns and removing id columns ****
      
Checking the date range

Splitting the data into training, validation, and test sets based on date

      
Encoding
   - columns to be encoded (OneHotEncoder) : 

   - Generating names for the new one-hot encoded features

   - Transforming the training, validation, and test sets

  - Dropping the original categorical features

Scaling Numerical Features
  - Create standard scaler
  - transform all categorical columns with standard scaler for each of X_train, X_valid, X_test data sets
  - 
Build model with MLFLOW, GRIDSearch, Hyperperameter tunning AND Evaluate Model performances
- Logistic Regression
- Random Forest
- XGBoost

**Truck delay application****

Step 1: Develop Streamlit applicaton (app.py) to create Truckdelay predictions as shown in above picture

connect to Hopsworks to get the final merge dataset

connect to mlflow model registry to get the model, encoder.pkl, scalar.pkl

write the python code to create streamlit application

    



