FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/telemedicina telemedicina
EXPOSE 8001
CMD ["python", "-m", "telemedicina.monitor.main"]
