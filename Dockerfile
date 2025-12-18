FROM python:3.10-slim

# Install system dependencies
# libsndfile1 for soundfile, ffmpeg for audio processing, build-essential for compiling some python packages
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port
EXPOSE 8080

# Environment variables
# PYTHONUNBUFFERED ensures logs are streamed to Cloud Logging
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
