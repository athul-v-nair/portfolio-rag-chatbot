# Use a small Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Set the working directory
WORKDIR /app

# Install system dependencies (lightweight)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# We copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only the necessary source code and data
COPY api/ api/
COPY src/ src/
COPY data/ data/
COPY .env .env

# Expose the app's port
EXPOSE 8000

# Command to run the application
# Using uvicorn directly for simplicity, but with host/port configuration
CMD ["sh", "-c", "uvicorn api.api:app --host 0.0.0.0 --port ${PORT}"]
