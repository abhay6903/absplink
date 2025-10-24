import os

# MinIO/S3 Configurations
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://65.1.6.222:30625")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "minio")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "minio123")
S3_BUCKET = os.environ.get("S3_BUCKET", "dedupedreports")

# Trino Default User
TRINO_DEFAULT_USER = os.environ.get("TRINO_DEFAULT_USER", "spark")