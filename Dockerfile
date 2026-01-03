# Server-friendly Streamlit image with Tesseract installed
FROM python:3.12-slim

# System deps: tesseract for OCR
RUN apt-get update \
  && apt-get install -y --no-install-recommends tesseract-ocr \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Streamlit defaults for container
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

EXPOSE 8501

# Pick the app you want to serve
CMD ["streamlit", "run", "horse.py", "--server.address=0.0.0.0", "--server.port=8501"]
