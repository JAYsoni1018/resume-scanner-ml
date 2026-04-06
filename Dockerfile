FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train model on build
RUN python src/train.py

EXPOSE 5000 8501

# Start Flask API
CMD ["python", "app/app.py"]
