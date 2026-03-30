FROM python:3.11-slim
WORKDIR /workspace
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY app/ ./app/
COPY src/ ./src/
COPY data/raw/ ./data/raw/
EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]