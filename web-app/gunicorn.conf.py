"""
Gunicorn configuration for the web app
"""

import os

# Server socket binding
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:5000")
backlog = 2048

# Worker processes
workers = int(os.getenv("GUNICORN_WORKERS", "1"))
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "sync")
threads = 1
worker_connections = 1000
timeout = 120
graceful_timeout = 30
keepalive = 2

# Server mechanics
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# The maximum number of requests a worker will process before restarting
# Helps prevent memory leaks
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "1000"))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "50"))

# Logging
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
accesslog = os.getenv("GUNICORN_ACCESS_LOG", "-")  # Log to stdout
errorlog = os.getenv("GUNICORN_ERROR_LOG", "-")   # Log to stderr
