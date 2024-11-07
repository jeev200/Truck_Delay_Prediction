FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt /app/

# Install system dependencies and update pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libpq-dev libcurl4-openssl-dev && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . /app

# Copy the MLruns directory to the appropriate location in the image


# Expose the port Streamlit will run on
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
