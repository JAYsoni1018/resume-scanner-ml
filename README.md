# Resume Role Classifier 🚀

A full-stack ML project that predicts job role categories from uploaded resumes with confidence scores.

## Features
- Upload PDF or DOCX resumes
- Predicts role: Android Developer, Data Scientist, Web Developer, etc. (25 categories)
- Shows confidence score (e.g., 93%)
- Flask REST API backend
- Streamlit UI frontend

## Project Structure
```
project-name/
├── README.md
├── requirements.txt
├── Dockerfile
├── config.yaml
├── data/
│   ├── README.md
│   └── dataset_link.txt
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
├── pipeline/
│   └── pipeline.py
├── models/
│   └── model_v1.pkl
├── app/
│   ├── app.py
│   └── schema.py
├── logs/
│   └── app.log
└── notebooks/
```

## Quick Start

### 1. Train the model
```bash
python src/train.py
```

### 2. Start Flask API
```bash
python app/app.py
```

### 3. Start Streamlit UI
```bash
streamlit run streamlit_app.py
```

## API Endpoints
- `POST /predict` — Upload resume file, returns predicted role + confidence
- `GET /health` — Health check
- `GET /categories` — List all supported role categories
