import os
import sys
from dataclasses import dataclass
from datetime import datetime

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    min_accuracy_threshold: float = 0.7
    confusion_matrix_path: str = os.path.join("artifacts", "confusion_matrix.png")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("⚙️  Starting model training pipeline for classification...")

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
                "Random Forest": {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20]
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 150],
                    'learning_rate': [0.1, 0.05]
                },
                "Logistic Regression": {
                    'C': [0.1, 1.0, 10.0]
                },
                "XGBoost": {
                    'n_estimators': [100, 150],
                    'learning_rate': [0.1, 0.05],
                    'max_depth': [3, 5, 7]
                },
                "Naive Bayes": {},
                "SVC": {
                    'C': [0.1, 1.0],
                    'kernel': ['linear', 'rbf']
                },
                "AdaBoost": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1, 1.0]
                },
            }

            best_model = None
            best_model_name = None
            best_score = 0
            best_report = {}
            model_report = {}

            for model_name, model in models.items():
                try:
                    logging.info(f"🔍 Training model: {model_name}")
                    model_params = params.get(model_name, {})
                    gs = GridSearchCV(model, model_params, cv=3, n_jobs=-1, verbose=0)
                    gs.fit(X_train, y_train)

                    y_pred = gs.predict(X_test)
                    test_acc = accuracy_score(y_test, y_pred)
                    model_report[model_name] = test_acc

                    if test_acc > best_score:
                        best_score = test_acc
                        best_model = gs.best_estimator_
                        best_model_name = model_name
                        best_report = classification_report(y_test, y_pred, output_dict=True)

                    logging.info(f"✅ {model_name} Accuracy: {test_acc:.4f}")
                except Exception as e:
                    logging.warning(f"⚠️ Skipping {model_name} due to error: {e}")
                    model_report[model_name] = 0

            if best_score < self.model_trainer_config.min_accuracy_threshold:
                raise CustomException("🚫 No model with acceptable accuracy was found", sys)

            logging.info(f"🏆 Best Model: {best_model_name} with Accuracy: {best_score:.4f}")
            logging.info(f"🧪 Classification Report for Best Model:\n{classification_report(y_test, best_model.predict(X_test))}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"💾 Best model saved to: {self.model_trainer_config.trained_model_file_path}")

            # Save confusion matrix
            try:
                ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
                plt.title(f'Confusion Matrix - {best_model_name}')
                plt.tight_layout()
                plt.savefig(self.model_trainer_config.confusion_matrix_path)
                logging.info(f"📊 Confusion matrix saved at: {self.model_trainer_config.confusion_matrix_path}")
            except Exception as e:
                logging.warning(f"⚠️ Failed to generate confusion matrix: {e}")

            # Return structured training result
            return {
                "best_model_name": best_model_name,
                "accuracy": best_score,
                "model_path": self.model_trainer_config.trained_model_file_path,
                "confusion_matrix_path": self.model_trainer_config.confusion_matrix_path,
                "classification_report": best_report
            }

        except Exception as e:
            raise CustomException(e, sys)
