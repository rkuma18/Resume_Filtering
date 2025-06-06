import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

def main():
    try:
        # Ensure artifacts directory exists
        os.makedirs("artifacts", exist_ok=True)
        
        # Data Ingestion
        logging.info("Starting data ingestion")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion completed. Train data: {train_data_path}, Test data: {test_data_path}")

        # Data Transformation
        logging.info("Starting data transformation")
        data_transformation = DataTransformation()
        X_train, X_test, y_train, y_test, preprocessor_path, label_encoder_path = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info(f"Data transformation completed. Preprocessor saved at: {preprocessor_path}")

        # Model Training
        logging.info("Starting model training")
        model_trainer = ModelTrainer()
        training_result = model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
        logging.info(f"Model training completed with accuracy: {training_result['accuracy']:.4f}")

    except Exception as e:
        logging.error(f"Error in training pipeline: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 