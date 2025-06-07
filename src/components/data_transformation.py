import sys
import os
import re
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_scaler_path: str = os.path.join("artifacts", "scaler.pkl")
    label_encoder_path: str = os.path.join("artifacts", "label_encoder.pkl")
    sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.model = SentenceTransformer(self.config.sentence_model_name)
        self.scaler = StandardScaler()
        self.skill_keywords = [
            "python", "sql", "excel", "machine learning", "deep learning",
            "power bi", "tableau", "data analysis", "project management", "aws", "azure"
        ]

    def clean_text(self, text):
        text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    def extract_structured_features(self, df):
        df["resume_length"] = df["Resume_str"].apply(lambda x: float(len(x.split())))
        for skill in self.skill_keywords:
            df[f"has_{skill.replace(' ', '_')}"] = df["Resume_str"].apply(lambda x: int(skill in x))
        return df

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df["Resume_str"] = train_df["Resume_str"].astype(str).apply(self.clean_text)
            test_df["Resume_str"] = test_df["Resume_str"].astype(str).apply(self.clean_text)

            train_embed = self.model.encode(train_df["Resume_str"].tolist(), show_progress_bar=True)
            test_embed = self.model.encode(test_df["Resume_str"].tolist(), show_progress_bar=True)

            train_df = self.extract_structured_features(train_df)
            test_df = self.extract_structured_features(test_df)

            struct_cols = ["resume_length"] + [f"has_{kw.replace(' ', '_')}" for kw in self.skill_keywords]
            train_struct = self.scaler.fit_transform(train_df[struct_cols])
            test_struct = self.scaler.transform(test_df[struct_cols])

            save_object(self.config.preprocessor_scaler_path, self.scaler)

            X_train = np.hstack([train_embed, train_struct])
            X_test = np.hstack([test_embed, test_struct])

            le = LabelEncoder()
            y_train = le.fit_transform(train_df["Category"])
            y_test = le.transform(test_df["Category"])
            save_object(self.config.label_encoder_path, le)

            return X_train, X_test, y_train, y_test, self.config.preprocessor_scaler_path, self.config.label_encoder_path

        except Exception as e:
            raise CustomException(e, sys)