import os
import sys
import pandas as pd
import logging
from src.exception import CustomException
from src.logger import logging

class DataIngestion:
    def __init__(self, data_path):
        self.data_path = data_path
        self.logger = logging()

    def load_data(self):
        try:
            # Load the CSV file into a Pandas dataframe
            data_df = pd.read_csv(self.data_path)

            # Shuffle the dataframe rows
            data_df = data_df.sample(frac=1).reset_index(drop=True)

            self.logger.info(f"Data loaded successfully from {self.data_path}")
            return data_df

        except Exception as e:
            self.logger.info(f"Failed to load data from {self.data_path}: {e}")
            raise CustomException(e, sys)

