"""
Gunicorn configuration for the web app
"""
import multiprocessing

# Server socket binding
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
workers = 1  # Start with 1 worker for this app
worker_class = "sync"
threads = 1
worker_connections = 1000
timeout = 120
graceful_timeout = 30
keepalive = 2

# Server mechanics
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190 