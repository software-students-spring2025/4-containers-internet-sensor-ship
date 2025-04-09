FROM python:3.8-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt Pipfile Pipfile.lock ./
# Ensure pip is up-to-date and install requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create .env file with defaults if none exists
RUN echo "SECRET_KEY=dev-secret-key-docker" > .env && \
    echo "MONGODB_URI=mongodb://mongodb:27017/" >> .env && \
    echo "MONGODB_DBNAME=cat_feeder" >> .env

# Copy the application code
COPY src/ ./src/
COPY templates/ ./templates/
COPY tests/ ./tests/
COPY wsgi.py ./
# Create empty static folder if it doesn't exist
RUN mkdir -p static

# Expose the port Gunicorn will run on
EXPOSE 5000

# Set environment variables for Flask
ENV PYTHONPATH=/app
ENV FLASK_APP=wsgi:application
ENV FLASK_ENV=production

# Add a healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Run the application using Gunicorn with worker timeout
# Point to wsgi.py instead of trying to import create_app directly
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "1", "--log-level", "debug", "wsgi:application"] 