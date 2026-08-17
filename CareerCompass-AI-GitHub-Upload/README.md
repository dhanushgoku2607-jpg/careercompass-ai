# CareerCompass AI

An AI/ML minor-project application that predicts placement likelihood and suggests practical skill improvements.

## Run locally

1. Install Python 3.10+.
2. Open a terminal in this folder and run `pip install -r requirements.txt`.
3. Train the model on the included dataset: `python train_model.py`.
4. Start the web application: `streamlit run app.py`.

## Project scope

- **Input:** CGPA, attendance, aptitude, technical/coding/communication scores, projects, internships, and certifications.
- **Output:** placement probability, profile category, and personalised action plan.
- **Model:** Random Forest classifier, evaluated with a hold-out test split.

## Team roles

| Member | Responsibility |
| --- | --- |
| 1 | Dataset validation, preprocessing, model training and metrics |
| 2 | Streamlit UI, prediction flow and dashboard |
| 3 | Testing, project report, PPT, demo and deployment notes |

## Submission checklist

The included `data/placementdata.csv` was downloaded from the [Kaggle Placement Prediction Dataset](https://www.kaggle.com/datasets/ruchikakumbhar/placement-prediction-dataset). Cite this source in your report and follow its licence/terms. Record your final accuracy, precision, recall and F1-score in the report. Do not claim that this dataset was collected from your college.
