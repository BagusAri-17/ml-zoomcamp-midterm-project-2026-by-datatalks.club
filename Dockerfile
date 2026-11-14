# Use official slim Python runtime as base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Ensure Python output is sent straight to terminal without buffering
ENV PYTHONUNBUFFERED=1

# Install project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy serialized model artifact and prediction service
COPY models/ ./models/
COPY predict.py .
COPY sample_students.json .

# Expose port for FastAPI service
EXPOSE 9696

# Run Uvicorn server
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]

