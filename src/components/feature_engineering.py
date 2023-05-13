import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer



class FeatureEngineering:
    def __init__(self, data_df):
        self.data_df = data_df
        self.logger = logging
    
    def tokenize_text(self):
        """
        Tokenize the text data.
        """
        tokenized_texts = []
        
        try:
            tokenized_texts = []
            for text in self.data_df['text']:
                tokens = word_tokenize(text)
                tokenized_texts.append(tokens)
            return tokenized_texts    
        
        except Exception as e:
            self.logger.info(e)
            raise CustomException(e, sys)
        
    def vectorize_text(self,tokenized_texts):
        """
        Vectorize the tokenized texts using TF-IDF.
        """
        try:
            vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
            vectorized_texts = vectorizer.fit_transform(tokenized_texts)
            return vectorized_texts, vectorizer    
        
        except Exception as e:
            self.logger.info(e)
            raise CustomException(e, sys)
        
   
    