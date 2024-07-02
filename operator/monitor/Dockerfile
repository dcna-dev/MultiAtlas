# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY datasource.py .

# Copy the monitor.py file into the container
COPY monitor.py .

# Set the command to run when the container starts
CMD ["python", "monitor.py"]
