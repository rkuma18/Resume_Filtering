# AI Resume Classifier

This is an intelligent resume classification pipeline that uses **BERT sentence embeddings** and **structured features** to predict the most suitable job category for a given resume. It's designed to help automate HR screening, job board categorization, or AI-based resume filtering systems.

---

## Features

* Accepts raw resume text and predicts the job category.
* Uses `sentence-transformers` (BERT) for semantic understanding.
* Adds structured resume features (e.g., resume length, presence of key skills).
* Trains and evaluates multiple models using GridSearchCV.
* Saves performance metrics (confusion matrix, classification report).
* Provides an easy-to-use **Streamlit Web Interface**.

---

## Project Structure

```
Resume_Filtering/
│
├── app.py                        ← Streamlit web app
├── main.py                       ← Main training pipeline
├── artifacts/                    ← Saved models, encoders, scalers, reports
├── data/                         ← Raw input data (train.csv, test.csv)
├── src/
│   ├── components/
│   │   ├── data_ingestion.py     ← Loads and splits the dataset
│   │   ├── data_transformation.py← Text + feature engineering
│   │   └── model_trainer.py      ← Trains & evaluates models
│   ├── pipeline/
│   │   └── predict_pipeline.py   ← Loads model, predicts from input
│   ├── utils.py                  ← Utility functions (save/load models, evaluate)
│   ├── logger.py                 ← Logs events during execution
│   └── exception.py              ← Custom exception handling
└── README.md
```

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/resume-classifier.git
   cd resume-classifier
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the following model is downloaded**

   * SentenceTransformer: `all-MiniLM-L6-v2`
     This will be auto-downloaded during training or prediction.

---

## Input Data Format

The input CSV files (`train.csv`, `test.csv`) must have the following columns:

| Resume\_str                      | Category     |
| -------------------------------- | ------------ |
| "Experienced in SQL, Excel..."   | "ACCOUNTANT" |
| "Worked in patient care unit..." | "HEALTHCARE" |

---

## How to Run

### Run the Training Pipeline

This will:

* Load and transform the dataset
* Train multiple models using `GridSearchCV`
* Save the best model + scaler + label encoder
* Generate reports

```bash
python main.py
```

---

### Launch Streamlit App

This allows you to paste any resume text and see the predicted job category.

```bash
streamlit run app.py
```

Then open the link shown in your terminal (usually `http://localhost:8501`).

---

## Sample Resume Inputs to Test

You can paste resumes like:

### IT Example:

```
Skilled software engineer with 4+ years of experience in backend development using Python and Django. Proficient in SQL, REST APIs, and deploying ML models on AWS. Worked with cross-functional teams in agile setups.
```

### Healthcare Example:

```
Registered nurse with 5 years of clinical experience in ICU units. Proficient in patient care, vital monitoring, and emergency protocols. Skilled in electronic medical record systems.
```

### Banking Example:

```
Finance professional with 6+ years in corporate banking. Managed credit risk assessments, trade finance, and treasury operations. Expertise in Excel and financial modeling.
```

---

## Evaluation Metrics (Sample)

Model performance is saved under `artifacts/`:

* `model.pkl`: Best model
* `scaler.pkl`: Scaler for structured features
* `label_encoder.pkl`: Label encoder for job categories
* `classification_report.txt`
* `confusion_matrix.png`

---

## Future Improvements

* Fine-tune BERT using HuggingFace for more accuracy
* Add resume parsing (PDF/docx) with spaCy or PyMuPDF
* Add SHAP or LIME for explainability
* REST API deployment using FastAPI

---

## Contact

If you have questions or suggestions, feel free to reach out at:
[kumarroushan.18@gmail.com](mailto:kumarroushan.18@gmail.com)
[Roushan Kumar](https://www.linkedin.com/in/rk0718/)

