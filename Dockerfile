FROM python:3.11-slim

WORKDIR /app
COPY . /app

# Install Java 17 and tools
RUN apt-get update && apt-get install -y default-jdk curl git && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gevent>=22.10.2

# Create persistent directories
RUN mkdir -p /app/outputs /app/tmp_checkpoints /app/spark-temp /app/spark-warehouse /app/logs

# Set env variables for Flask + Spark
ENV OUTPUTS_DIR=/app/outputs \
    CHECKPOINT_DIR=/app/tmp_checkpoints \
    SPARK_LOCAL_DIRS=/app/spark-temp \
    SPARK_WAREHOUSE_DIR=/app/spark-warehouse \
    LOG_DIR=/app/logs \
    PYTHONUNBUFFERED=1

EXPOSE 5000 4040

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--worker-class", "gevent", "--timeout", "3600", "apapp:app"]
