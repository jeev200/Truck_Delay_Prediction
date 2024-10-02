import sys
import os
import traceback

# Adjust the path according to your directory structure
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_directory)

# Importing pipeline stages from their respective modules
from pipelines.stage_01_data_ingestion import DataIngestionPipeline
from pipelines.stage_02_data_cleaning import DataIngestionAndCleaning
from pipelines.stage_03_data_merging import DataProcessingPipeline

def run_pipeline(stage_name, pipeline):
    try:
        print(f">>>>>> Stage {stage_name} started <<<<<<")
        pipeline.main()
        print(f">>>>>> Stage {stage_name} completed <<<<<<\n\nx==========x\n")
    except Exception as e:
        print(f"An error occurred during {stage_name}: {e}")
        traceback.print_exc() 
        raise

if __name__ == "__main__":
    run_pipeline("Data Ingestion", DataIngestionPipeline())
    run_pipeline("Data Cleaning", DataIngestionAndCleaning())
    run_pipeline("Data Preparation", DataProcessingPipeline())
