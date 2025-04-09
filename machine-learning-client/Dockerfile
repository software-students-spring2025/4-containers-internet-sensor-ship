FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV and downloading
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

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

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make sure the Flask app is the entrypoint
CMD ["python", "app.py"] 