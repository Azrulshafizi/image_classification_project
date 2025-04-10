# gunicorn_config.py - Configuration for Gunicorn WSGI server

import os
import multiprocessing

# Server socket configuration
bind = "0.0.0.0:" + os.environ.get("PORT", "5000")

# Worker processes
# For CPU-bound applications like YOLO, best to use 1 worker on Render free tier
workers = int(os.environ.get("WEB_CONCURRENCY", 1))

# Timeout for worker processes (in seconds)
# This needs to be high enough for YOLO inference to complete
timeout = 120

# Worker class - use sync for CPU-intensive tasks
worker_class = "sync"

# Adjust threads per worker - less is better for CPU-intensive work
threads = 1

# Max requests before worker restart - helps prevent memory leaks
max_requests = 100
max_requests_jitter = 10

# Logging
accesslog = "-"  # Output to stdout
errorlog = "-"   # Output to stderr
loglevel = "info"

# Limit request line to prevent DoS
limit_request_line = 4094

# Preload application - reduces memory usage
preload_app = True

# Initialize application - runs before workers are forked
def on_starting(server):
    print("Gunicorn server is starting...")

# Pre-fork worker
def pre_fork(server, worker):
    # Perform any pre-fork tasks like model initialization
    pass

# Post-fork worker
def post_fork(server, worker):
    # Import only here to prevent memory issues during fork
    try:
        import gc
        # Aggressive garbage collection
        gc.collect()
    except ImportError:
        pass

# Worker exit - clean up resources
def worker_exit(server, worker):
    try:
        import gc
        import torch
        # Clean up CUDA memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Force garbage collection
        gc.collect()
    except ImportError:
        pass

# Pre-execution hooks - before processing a request
def pre_request(worker, req):
    worker.start_time = worker.age