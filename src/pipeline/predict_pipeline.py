import os
import sys
import re
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        # Define paths
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.label_encoder_path = os.path.join("artifacts", "label_encoder.pkl")

    def clean_resume(self, text: str) -> str:
        try:
            text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.IGNORECASE)
            text = re.sub(r'\bRT\b|\bcc\b', ' ', text)
            text = re.sub(r'#(\w+)', r'\1', text)
            text = re.sub(r'@\w+', ' ', text)
            text = re.sub(r'[^\x00-\x7f]', ' ', text)
            text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text.lower()
        except Exception as e:
            raise CustomException(f"Error during text cleaning: {e}", sys)

    def predict(self, features: pd.DataFrame):
        try:
            # Load components
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)
            label_encoder = load_object(self.label_encoder_path)

            # Clean resume text
            features["Resume_str"] = features["Resume_str"].apply(self.clean_resume)

            # Transform text using preprocessor (TF-IDF)
            data_transformed = preprocessor.transform(features["Resume_str"])

            # Predict category
            predictions = model.predict(data_transformed)

            # Decode label
            predicted_labels = label_encoder.inverse_transform(predictions)

            return predicted_labels

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, resume_text: str):
        self.resume_text = resume_text

    def get_data_as_data_frame(self):
        try:
            return pd.DataFrame({"Resume_str": [self.resume_text]})
        except Exception as e:
            raise CustomException(e, sys)
