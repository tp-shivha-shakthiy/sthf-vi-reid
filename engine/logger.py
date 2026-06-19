import os
from datetime import datetime


class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        log_file = datetime.now().strftime("train_%Y%m%d_%H%M%S.log")
        self.log_path = os.path.join(log_dir, log_file)

    def log(self, message, end="\n"):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        full = f"{timestamp} {message}"
        print(full, end=end)
        with open(self.log_path, "a") as f:
            f.write(full + "\n")
