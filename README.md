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

## Quick Start

### 1. Train the model

```bash
python train.py
```

### 2. Start Flask API

```bash
python app.py
```

### 3. Start Streamlit UI

```bash
streamlit run streamlit_app.py
```

## API Endpoints

- `POST /predict` — Upload resume file, returns predicted role + confidence
- `GET /health` — Health check
