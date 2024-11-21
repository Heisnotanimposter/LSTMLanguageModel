# logging_monitoring.py

import logging
import time
from prometheus_client import start_http_server, Summary, Counter
from functools import wraps

# Configure logging
logging.basicConfig(
    filename='application.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(module)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Prometheus metrics
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
RECOMMENDATION_COUNT = Counter('recommendation_requests_total', 'Total recommendation requests')

def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with REQUEST_TIME.time():
            result = func(*args, **kwargs)
        return result
    return wrapper

def log_user_interaction(user_id, video_id, action):
    logging.info(f"User {user_id} performed {action} on video {video_id}")

@log_execution_time
def process_request(user_id, video_id):
    # Simulate processing
    time.sleep(0.1)
    log_user_interaction(user_id, video_id, 'watch')

def start_metrics_server():
    # Start up the server to expose the metrics.
    start_http_server(8000)
    logging.info("Prometheus metrics server started on port 8000")

if __name__ == "__main__":
    start_metrics_server()
    # Simulate some requests
    for i in range(10):
        process_request(user_id=i, video_id=100 + i)
        RECOMMENDATION_COUNT.inc()
        time.sleep(1)