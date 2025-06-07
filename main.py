import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException

def main():
    try:
        logging.info("🚀 Starting pipeline")
        os.makedirs("artifacts", exist_ok=True)

        di = DataIngestion()
        train_path, test_path = di.initiate_data_ingestion()

        dt = DataTransformation()
        X_train, X_test, y_train, y_test, _, _ = dt.initiate_data_transformation(train_path, test_path)

        mt = ModelTrainer()
        result = mt.initiate_model_trainer(X_train, y_train, X_test, y_test)

        logging.info(f"✅ Model Saved At: {result['model_path']}")

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()