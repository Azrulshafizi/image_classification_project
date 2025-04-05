# gunicorn_config.py

# Use only a single worker to minimize memory usage
workers = 1

# Use threads instead of additional processes
threads = 2

# Use thread-based worker
worker_class = 'gthread'

# Increase timeout for model loading and prediction (in seconds)
timeout = 300

# Restart workers periodically to help with memory management
max_requests = 10
max_requests_jitter = 3

# Preload app to ensure model is loaded at startup rather than on first request
preload_app = False  # Set to False to lazy load the model

# Log level
loglevel = 'info'

# Bind to port from environment variable or default to 8080
bind = "0.0.0.0:" + os.environ.get("PORT", "8080")

# Print memory usage on startup
def on_starting(server):
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mb = process.memory_info().rss / 1024 / 1024
        print(f"[Gunicorn] Initial memory usage: {mb:.2f} MB")
    except ImportError:
        print("[Gunicorn] Could not log memory (psutil not available)")