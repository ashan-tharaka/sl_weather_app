# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose the port Flask runs on
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]