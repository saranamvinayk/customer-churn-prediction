# 1. Start with a lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements first
COPY requirements.txt .

# 4. Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your source code and model artifacts
COPY src/ ./src/
COPY model.joblib .
COPY scaler.joblib .

# 6. Expose the port the API will run on
EXPOSE 8000

# 7. Command to run the FastAPI server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
