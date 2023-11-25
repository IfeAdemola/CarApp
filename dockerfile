# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY scripts/__init__.py scripts/__init__.py   
COPY scripts/inference.py scripts/inference.py
COPY scripts/preprocessing.py scripts/preprocessing.py
COPY app/app.py .
COPY app/requirements.txt .

# . . copies all the files in the same folder as this dockerfile

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit will run on (default is 8501)
EXPOSE 8501

# Define the command to run your Streamlit app when the container starts
CMD ["streamlit", "run", "app.py"]
