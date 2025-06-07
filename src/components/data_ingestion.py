import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', "data.csv")
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    source_data_path: str = os.path.join("notebook", "data", "resume", "Resume", "Resume.csv")
    test_size: float = 0.2
    random_state: int = 42

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("📥 Reading source data")
        try:
            df = pd.read_csv(self.config.source_data_path)
            df = df[["Resume_str", "Category"]].dropna()
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)

            train_df, test_df = train_test_split(
                df, test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=df['Category']
            )
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)