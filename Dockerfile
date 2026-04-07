# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY env/ env/

# Copy app, configs, and inference script
COPY server/ server/
COPY openenv.yaml .
COPY inference.py .
COPY llm_client.py .

# Default command matches HF Spaces execution (uvicorn on port 7860)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
