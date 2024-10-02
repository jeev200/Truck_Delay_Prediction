import os
import sys
import configparser

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_directory)

from src.utils.config_utils import *
from src.utils.db_utlis import * 

from src.components.data_ingestion import PostgreSQLIngestion

class DataIngestionPipeline:
    
    def __init__(self):
        self.config = read_config()

    def main(self):
        print("Starting data ingestion...")

        try:
            
            db_config = get_db_config(self.config)
            source_url = self.config['DATA']['source_url'].strip()
            ingestion = PostgreSQLIngestion(db_config)
            engine = connect_to_db(db_config)
            csv_files = ingestion.fetch_github_raw_files(source_url)

           
            for file_url in csv_files:
                table_name = file_url.split('/')[-1].replace('.csv', '')
                ingestion.upload_to_postgres(engine, file_url, table_name)

        except Exception as e:
            print(f"An error occurred: {e}")
            raise e

        finally:
            close_db_connection(engine)


if __name__ == '__main__':
    try:
        STAGE_NAME = "Data Ingestion"
        print(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        pipeline = DataIngestionPipeline()
        pipeline.main()
        print(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
