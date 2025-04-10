FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV and downloading
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY Pipfile ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install flask==2.2.3 \
                werkzeug==2.2.3 \
                numpy==1.26.4 \
                opencv-python==4.8.0.76 \
                python-dotenv==1.0.0 \
                requests==2.31.0 \
                pymongo==4.5.0 \
                pytz==2023.3

# Create models directory
RUN mkdir -p models/cascades

# Download OpenCV's pre-trained Haar cascade for face detection (as example)
RUN wget -O models/cascades/haarcascade_frontalface_default.xml \
    https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

# Download a cat detector cascade from opencv-extra
RUN wget -O models/cascades/haarcascade_frontalcatface.xml \
    https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalcatface.xml

# Download the extended version too
RUN wget -O models/cascades/haarcascade_frontalcatface_extended.xml \
    https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalcatface_extended.xml

# Copy the application code
COPY . .

# Set Python path to include src directory
ENV PYTHONPATH=/app

# Make sure the Flask app is the entrypoint
CMD ["python", "src/app.py"] 