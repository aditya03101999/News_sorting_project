import sys
import re
import pandas as pd
import joblib
import numpy as np
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import logging
from src.exception import CustomException




class FeatureEngineering:
    def __init__(self):
        self.logger = logging
                        
    def stopwords_removal(self,df):
        try:
            # Tokenizing the words
            stop_words = set(stopwords.words('english'))
            tokenized_texts = []
            for text in df['Text']:
                tokens = word_tokenize(text)
                tokenized_texts.append(tokens)
            # Remove stop words
            texts = [] 
            for text in tokenized_texts:
                words= []
                for word in text:
                    if word not in stop_words:
                       words.append(word)
                texts.append(words)
            return texts
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
        
   
    