FROM python:3.10-slim
WORKDIR /app
ENV PYTHONPATH=/app/shared
COPY services/analyzer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY services/analyzer/app.py .
CMD ["python", "app.py"]
