# Resume Role Classifier 🚀

A full-stack ML project that predicts job role categories from uploaded resumes with confidence scores.

## Features

- Upload PDF or DOCX or Text resumes
- Predicts role: Android Developer, Data Scientist, Web Developer, etc. (20 categories)
- Shows confidence score (e.g., 93%)
- Flask REST API backend
- Streamlit UI frontend

## Project Structure

```
project-name/
├── README.md
├── requirements.txt
├── app.py
├── train.py
├── streamlit_app.py
├── Dockerfile
├── .dockerignore
├── config.yaml
├── .env
├── .gitignore
├── data/
│   ├── README.md
│   └── processed_resume_dataset.txt
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── features.py
│   ├── model_trainer.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
├── models/
│   └── model_v1.pkl
|    └── feature_artifacts.pkl
├── app/
│    └── schema.py
├── logs/
│   └── app.log
└── notebooks/
```

## Dataset

    This project uses a combination of two publicly available resume datasets:

    Dataset 1: https://www.kaggle.com/datasets/haidermaseeh/resume-dataset
    Dataset 2: https://www.kaggle.com/datasets/arunsaini0906/resume-screening-dataset-for-nlp-and-ml




    Both datasets were merged to create a unified dataset for training the model.

    🔧 Data Preparation
    From the combined dataset, only two relevant columns were extracted:
    resume → raw resume text
    category → job role label

    Since the dataset contained a large number of categories, it was highly imbalanced and sparse.

    🎯 Category Selection
    To improve model performance and reduce noise:
    Only the top 20 most frequent categories were selected
    Remaining categories were excluded from training

    This helped:

    Reduce class imbalance
    Improve model accuracy
    Ensure sufficient data per category

## Quick Start

### 1. Create virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

### 3. Start Flask API

```bash
python app.py
```

### 4. Start Streamlit UI

```bash
streamlit run streamlit_app.py
```

## API Endpoints

- `POST /predict` — Upload resume file, returns predicted role + confidence
- `GET /health` — Health check
