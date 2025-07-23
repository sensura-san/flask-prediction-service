# Use a lightweight Python image
FROM python:3.9-slim

# Install system dependencies required by LightGBM
RUN apt-get update && apt-get install -y \
    build-essential \
    libboost-all-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libgmp-dev \
    libhdf5-dev \
    cmake

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . /app

# Expose the port that the Flask app will run on
EXPOSE 5000

# Command to run the Flask application using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
