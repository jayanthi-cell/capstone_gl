FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --timeout=1000 --retries 10 -r requirements.txt

COPY app.py .
COPY preprocessing.py .
COPY pneumonia_detection_resnet50_v1_0.keras 

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
