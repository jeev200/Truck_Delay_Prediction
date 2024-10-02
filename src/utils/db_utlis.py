import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

def connect_to_db(db_config):
    try:
        engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        )
        print("Database connection successful.")
        return engine
    except Exception as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        raise

def close_db_connection(engine):
    if engine:
        engine.dispose()
        print("Database connection closed.")
    else:
        print("No active database connection to close.")

import pandas as pd

def fetch_all_data(engine):
  
    if engine is None:
        raise ValueError("No database engine provided. Please pass a valid SQLAlchemy engine.")

    query = "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE';"
    table_names = pd.read_sql_query(query, engine)['table_name'].tolist()
    dataframes = {}
    for table in table_names:
        dataframes[table] = pd.read_sql_table(table, engine)
        
    return dataframes


def replace_cleaned_data(cleaned_data, engine):
   
    try:
        for table_name, cleaned_df in cleaned_data.items():
            cleaned_df.to_sql(table_name, engine, if_exists='replace', index=False)
            print(f"Replaced data in the table {table_name}")
    except SQLAlchemyError as e:
        print(f"An error occurred while replacing data for table {table_name}: {e}")
