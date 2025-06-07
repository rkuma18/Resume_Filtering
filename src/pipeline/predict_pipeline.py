## 6. `predict_pipeline.py`

import re
import pandas as pd

class CustomData:
    def __init__(self, resume_text: str):
        self.resume_text = resume_text

    def get_data_as_data_frame(self):
        return pd.DataFrame({"Resume_str": [self.resume_text]})

import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from src.utils import load_object

@dataclass
class PredictPipelineConfig:
    model_path = "artifacts/model.pkl"
    scaler_path = "artifacts/scaler.pkl"
    label_encoder_path = "artifacts/label_encoder.pkl"
    sentence_model_name = "sentence-transformers/all-MiniLM-L6-v2"

class PredictPipeline:
    def __init__(self):
        self.config = PredictPipelineConfig()
        self.model = load_object(self.config.model_path)
        self.scaler = load_object(self.config.scaler_path)
        self.label_encoder = load_object(self.config.label_encoder_path)
        self.embedder = SentenceTransformer(self.config.sentence_model_name)
        self.skill_keywords = [
            "python", "sql", "excel", "machine learning", "deep learning",
            "power bi", "tableau", "data analysis", "project management", "aws", "azure"
        ]

    def clean_text(self, text):
        text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    def extract_features(self, text):
        text = self.clean_text(text)
        embed = self.embedder.encode([text])[0]
        struct = [float(len(text.split()))] + [int(skill in text) for skill in self.skill_keywords]
        struct_scaled = self.scaler.transform([struct])[0]
        return np.hstack([embed, struct_scaled]).reshape(1, -1)

    def predict(self, resume_text):
        features = self.extract_features(resume_text)
        pred_encoded = self.model.predict(features)
        return self.label_encoder.inverse_transform(pred_encoded)[0]

class CustomData:
    def __init__(self, resume_text: str):
        self.resume_text = resume_text

    def get_data_as_data_frame(self):
        return pd.DataFrame({"Resume_str": [self.resume_text]})
