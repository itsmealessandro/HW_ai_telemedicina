FROM python:3.10-slim
WORKDIR /app
ENV PYTHONPATH=/app/shared
COPY services/executor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY services/executor/app.py services/executor/ws_manager.py ./
CMD ["python", "app.py"]
