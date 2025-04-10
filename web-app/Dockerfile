FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY Pipfile ./
RUN pip install --upgrade pip && \
    pip install flask==2.2.3 \
                flask-login==0.6.2 \
                werkzeug==2.2.3 \
                python-dotenv==1.0.0 \
                requests==2.31.0 \
                pytz==2023.3 \
                gunicorn==21.2.0 \
                pymongo==4.5.0

# Create .env file with defaults if none exists
RUN echo "SECRET_KEY=dev-secret-key-docker" > .env && \
    echo "MONGODB_URI=mongodb://mongodb:27017/" >> .env && \
    echo "MONGODB_DBNAME=cat_feeder" >> .env

# Copy the application code
COPY src/ ./src/
COPY templates/ ./templates/
COPY tests/ ./tests/
COPY wsgi.py ./
COPY gunicorn.conf.py ./
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
CMD ["gunicorn", "--config", "/app/gunicorn.conf.py", "wsgi:application"] 