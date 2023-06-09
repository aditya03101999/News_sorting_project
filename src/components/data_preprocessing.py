import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import LabelEncoder

class DataPreprocessing:
    def __init__(self, data_df):
        self.data_df = data_df
        self.logger = logging

    def encode_labels(self):
        try:
            # Remove rows with missing values
            self.data_df.dropna(inplace=True)
            
            # Encode the labels using label encoder
            label_encoder = LabelEncoder()
            self.data_df['Category_ID'] = label_encoder.fit_transform(self.data_df['category'])
            return self.data_df, label_encoder
        
        except Exception as e:
            self.logger.info(e)
            raise CustomException(e, sys)
