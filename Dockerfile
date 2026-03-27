# Use Python 3.10 slim as base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn httpx

# Copy the application code
COPY . .

# Expose port 8080
EXPOSE 8080

# Command to run the application
# We use the module path 'graph_rag.api:app'
CMD ["uvicorn", "graph_rag.api:app", "--host", "0.0.0.0", "--port", "8080"]
