# Use a slim Python 3.10 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt stopwords

# Copy application files
COPY . .

# Set environment variable for NLTK
ENV NLTK_DATA=/usr/local/share/nltk_data

# Expose the port Cloud Run expects
EXPOSE 8080

# Set environment variable for Flask
ENV PORT=8080

# Start the Flask app
CMD exec python lol.py