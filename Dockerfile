FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install prebuilt wheels for scientific packages first
RUN pip install --no-cache-dir --only-binary=:all: \
    numpy==1.25.2 \
    pandas==2.1.1 \
    scipy==1.11.4

# Then install the rest of the requirements
RUN pip install --no-cache-dir --upgrade -r requirements.txt

ENV APP_HOME=/root
WORKDIR $APP_HOME
COPY /app $APP_HOME/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]