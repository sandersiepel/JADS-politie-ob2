from collections import deque
from datetime import datetime

log_messages = deque(maxlen=5)  
    
# Function to add a new log message
def add_log_message(message):
    now = datetime.now().strftime("%H:%M:%S") + ": "
    log_messages.append(now + message)