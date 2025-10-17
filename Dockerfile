# Step 1: Start with a Python base image
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy the application and model files
COPY ./app.py /app/app.py
COPY ./models/ /app/models/

# Step 5: Expose the port the app will run on
EXPOSE 80

# Step 6: Define the command to run the API server
# This starts the uvicorn server, making the API accessible.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]