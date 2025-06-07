import os
import sys
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import json
import time

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    min_accuracy_threshold: float = 0.4
    confusion_matrix_path: str = os.path.join("artifacts", "confusion_matrix.png")
    classification_report_path: str = os.path.join("artifacts", "classification_report.txt")
    classification_report_json: str = os.path.join("artifacts", "classification_report.json")
    label_encoder_path: str = os.path.join("artifacts", "label_encoder.pkl")
    use_grid_search: bool = True

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("🚀 Starting model training...")

            models = {
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "XGBoost": XGBClassifier(eval_metric='mlogloss', use_label_encoder=False),
                "Naive Bayes": MultinomialNB(),
                "SVC": SVC(),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {
                "Random Forest": {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
                "Gradient Boosting": {'n_estimators': [100, 150], 'learning_rate': [0.1, 0.05]},
                "Logistic Regression": {'C': [0.1, 1.0, 10.0]},
                "XGBoost": {'n_estimators': [100, 150], 'learning_rate': [0.1, 0.05], 'max_depth': [3, 5, 7]},
                "Naive Bayes": {},
                "SVC": {'C': [0.1, 1.0], 'kernel': ['linear', 'rbf']},
                "AdaBoost": {'n_estimators': [50, 100], 'learning_rate': [0.1, 1.0]},
            }

            best_model = None
            best_model_name = None
            best_score = 0
            best_y_pred = None
            model_report = {}

            for model_name, model in models.items():
                try:
                    logging.info(f"🔍 Training {model_name}")
                    start = time.time()

                    model_params = params.get(model_name, {})
                    if self.config.use_grid_search and model_params:
                        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                        gs = GridSearchCV(model, model_params, cv=cv, n_jobs=-1, verbose=0)
                        gs.fit(X_train, y_train)
                        model_fitted = gs.best_estimator_
                    else:
                        model.fit(X_train, y_train)
                        model_fitted = model

                    duration = time.time() - start
                    y_pred = model_fitted.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                    model_report[model_name] = {
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1_score": f1,
                        "train_time_sec": duration
                    }

                    logging.info(f"✅ {model_name} Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Time: {duration:.2f}s")

                    if acc > best_score:
                        best_score = acc
                        best_model = model_fitted
                        best_model_name = model_name
                        best_y_pred = y_pred

                except Exception as e:
                    logging.warning(f"⚠️ Skipping {model_name} due to error: {e}")
                    model_report[model_name] = {
                        "accuracy": 0,
                        "precision": 0,
                        "recall": 0,
                        "f1_score": 0,
                        "train_time_sec": 0
                    }

            for name, metrics in model_report.items():
                logging.info(f"📊 {name} → Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

            if best_model is None or best_score < self.config.min_accuracy_threshold:
                logging.error("❌ No model met the minimum accuracy threshold")
                raise CustomException("🚫 No model met the minimum accuracy threshold", sys)

            label_encoder = load_object(self.config.label_encoder_path)
            decoded_preds = label_encoder.inverse_transform(best_y_pred)
            decoded_true = label_encoder.inverse_transform(y_test)

            final_report = classification_report(decoded_true, decoded_preds)
            report_dict = classification_report(decoded_true, decoded_preds, output_dict=True)

            with open(self.config.classification_report_path, "w") as f:
                f.write(final_report)
            with open(self.config.classification_report_json, "w") as f:
                json.dump(report_dict, f, indent=2)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            versioned_model_path = f"artifacts/model_{best_model_name}_{timestamp}.pkl"
            save_object(self.config.trained_model_file_path, best_model)
            save_object(versioned_model_path, best_model)
            logging.info(f"💾 Best model saved at: {versioned_model_path}")

            try:
                ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
                plt.title(f'Confusion Matrix - {best_model_name}')
                plt.tight_layout()
                plt.savefig(self.config.confusion_matrix_path)
                logging.info(f"📊 Confusion matrix saved at: {self.config.confusion_matrix_path}")
            except Exception as e:
                logging.warning(f"⚠️ Failed to generate confusion matrix: {e}")

            return {
                "best_model_name": best_model_name,
                "accuracy": best_score,
                "model_path": versioned_model_path,
                "confusion_matrix_path": self.config.confusion_matrix_path,
                "classification_report_path": self.config.classification_report_path,
                "classification_report_json": self.config.classification_report_json
            }

        except Exception as e:
            raise CustomException(e, sys)
