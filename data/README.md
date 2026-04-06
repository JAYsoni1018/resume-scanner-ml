# Data

This folder stores the resume dataset used for training the role classifier.

## Dataset
- **Source**: UpdatedResumeDataSet.csv
- **Records**: ~2400 labeled resumes
- **Labels**: 25 job role categories
- **Columns**: `Category`, `Resume`

## Usage
Place `UpdatedResumeDataSet.csv` in this folder, then run:
```bash
python src/train.py
```
