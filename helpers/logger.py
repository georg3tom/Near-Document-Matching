import os


class Logger:
    def __init__(self):
        self.enabled = bool(os.getenv("DEBUG"))

    def log(self, *args, **kwargs):
        if self.enabled:
            print(*args, **kwargs)
