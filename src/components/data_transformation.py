import sys
from dataclasses import dataclass
import os
import re
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    label_encoder_path = os.path.join("artifacts", "label_encoder.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def clean_resume(self, text: str) -> str:
        """
        Clean the resume text using regex: remove URLs, mentions, special characters, etc.
        """
        try:
            text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.IGNORECASE)
            text = re.sub(r'\bRT\b|\bcc\b', ' ', text)
            text = re.sub(r'#(\w+)', r'\1', text)
            text = re.sub(r'@\w+', ' ', text)
            text = re.sub(r'[^\x00-\x7f]', ' ', text)
            text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            text = text.lower()
            return text
        except Exception as e:
            raise CustomException(f"Error during text cleaning: {e}", sys)

    def get_data_transformer_object(self):
        """
        Returns a TF-IDF vectorizer pipeline for text feature extraction.
        """
        try:
            text_pipeline = Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer(
                        max_features=5000,
                        stop_words='english',
                        strip_accents='unicode'
                    ))
                ]
            )
            return text_pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Loading training and test data.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Validate required columns
            required_columns = {"ID", "Resume_html", "Resume_str", "Category"}
            if not required_columns.issubset(set(train_df.columns)) or not required_columns.issubset(set(test_df.columns)):
                raise CustomException(f"Missing required columns: {required_columns}", sys)

            # Drop unused columns
            train_df.drop(columns=["ID", "Resume_html"], inplace=True)
            test_df.drop(columns=["ID", "Resume_html"], inplace=True)

            # Clean Resume text
            train_df["Resume_str"] = train_df["Resume_str"].apply(self.clean_resume)
            test_df["Resume_str"] = test_df["Resume_str"].apply(self.clean_resume)

            logging.info("Resume text cleaned successfully.")

            # Separate features and target
            X_train = train_df["Resume_str"]
            y_train = train_df["Category"]
            X_test = test_df["Resume_str"]
            y_test = test_df["Category"]

            # Show class distribution for reference
            logging.info(f"Training label distribution: {dict(Counter(y_train))}")

            # Encode target labels
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)

            logging.info(f"Target labels encoded. Classes: {list(label_encoder.classes_)}")

            # TF-IDF vectorization
            preprocessing_obj = self.get_data_transformer_object()
            X_train_tfidf = preprocessing_obj.fit_transform(X_train)
            X_test_tfidf = preprocessing_obj.transform(X_test)

            logging.info("TF-IDF transformation completed.")

            # Save TF-IDF vectorizer
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info(f"TF-IDF vectorizer saved at: {self.data_transformation_config.preprocessor_obj_file_path}")

            # Save label encoder
            save_object(
                file_path=self.data_transformation_config.label_encoder_path,
                obj=label_encoder
            )
            logging.info(f"Label encoder saved at: {self.data_transformation_config.label_encoder_path}")

            # Return everything needed for training
            return (
                X_train_tfidf,
                X_test_tfidf,
                y_train_encoded,
                y_test_encoded,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.label_encoder_path
            )

        except Exception as e:
            raise CustomException(e, sys)
