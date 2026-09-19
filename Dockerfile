# Use lightweight official Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies without caching to keep the image lean
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set default entrypoint to run your agent
CMD ["python", "agent.py"]