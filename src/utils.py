import os
import sys
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save any Python object (model, preprocessor, label encoder, etc.) using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(f"Error saving object at {file_path}: {e}", sys)


def load_object(file_path):
    """
    Load a pickled Python object from file.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(f"Error loading object from {file_path}: {e}", sys)


def evaluate_classification_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple classification models with hyperparameter tuning.
    Returns a dictionary of model names and their test accuracy.
    """
    try:
        report = {}

        for model_name, model in models.items():
            try:
                params = param.get(model_name, {})
                print(f"🔍 Tuning {model_name}...")
                gs = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=0)
                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_
                y_pred = best_model.predict(X_test)
                test_acc = accuracy_score(y_test, y_pred)

                report[model_name] = test_acc
                print(f"✅ {model_name} Accuracy: {test_acc:.4f}")
            except Exception as e:
                print(f"⚠️ Error evaluating {model_name}: {e}")
                report[model_name] = 0

        return report

    except Exception as e:
        raise CustomException(f"Error during model evaluation: {e}", sys)
