FROM python:3.11-slim

# Set a working directory
WORKDIR /app

# Copy requirement file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ai_personalized_learning_assistant/ ai_personalized_learning_assistant/
COPY docker/entrypoint.sh /app/entrypoint.sh

# Expose the port used by uvicorn
EXPOSE 8080

# Entrypoint for running the FastAPI server
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
