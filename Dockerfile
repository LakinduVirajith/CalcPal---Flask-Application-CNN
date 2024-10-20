# Use an appropriate base image with Python 3.11
FROM python:3.11

# Set the working directory inside the container to /app
# This directory will be the root for all subsequent commands
WORKDIR /app

# Copy the application files from the host machine to the working directory in the container
COPY app.py .
COPY requirements.txt .

# Copy the model files into the /app/models directory inside the container
# This allows the application to access the model files for predictions
COPY models/ /app/models/

# Install Python dependencies specified in requirements.txt
# --no-cache-dir option prevents caching of package installations to keep the image slim
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5002 for the Flask application to be accessible outside the container
EXPOSE 5002

# Define the command to run the Flask application
# This will be executed when the container starts
CMD ["python", "app.py"]
