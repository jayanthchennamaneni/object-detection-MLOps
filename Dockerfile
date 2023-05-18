# Use a minimal Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the requirements.txt file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Expose the application port
EXPOSE 8000

# Start the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
