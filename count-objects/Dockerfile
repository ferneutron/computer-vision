# Use the Python 3.9 slim image as the base image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install dependencies and then remove the cache
RUN apt-get update && pip install --trusted-host pypi.python.org -r requirements.txt

# Create a user called 'app' with sudo permissions
RUN useradd -m app && echo "app ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Switch to the 'app' user
USER app

# Set the environment variables from the .env file
CMD ["streamlit", "run", "app.py"]