FROM python:3.8-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
# Ensure pip is up-to-date and install requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# Make sure the src directory and templates are copied
COPY src/ ./src
COPY templates/ ./templates
# Copy other necessary files if any (e.g., static files, instance config)
# COPY instance/ ./instance

# Expose the port Gunicorn will run on
EXPOSE 5000

# Set environment variables (optional, can also be set at runtime)
# ENV FLASK_APP=src:create_app
# ENV FLASK_ENV=production # Or development

# Run the application using Gunicorn
# Bind to 0.0.0.0 to be accessible from outside the container
# Use a sensible number of workers (e.g., based on CPU cores)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src:create_app()"] 