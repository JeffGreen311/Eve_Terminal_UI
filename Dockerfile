# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     python3-dev     libpq-dev     && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the port Railway will use
EXPOSE 8080

# Start Eve's Terminal UI
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "60", "web_run:app"]
