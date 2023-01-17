# Logger setup

---

## Logger common setup

```python
import logging
import os

def get_logger(name):
    logger = logging.getLogger(name)
    logger.propagate = False

    # terminal output
    console = logging.StreamHandler()
    simple_formatter = logging.Formatter('%(name)s - %(message)s')
    console.setFormatter(simple_formatter)
    logger.addHandler(console)

    # full logging syntax in the log file
    log_file_path = os.getenv("LOG_FILE")
    if log_file_path:
        extended_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(log_file_path)
        fh.setFormatter(extended_formatter)
        logger.addHandler(fh)

    # it defines which level to output
    log_level = os.getenv("LOG_LEVEL")
    if log_level is None:
        raise Exception("Global Env: LOG_LEVEL is not set, did you forget to set env file or env variables?")
    logger.setLevel(log_level)
    return logger

# can be used like this:
logger = get_logger(__name__)
logger.info("hi")
```

## Coloring logging

```python
import logging
import os

# Define a custom formatter
class CustomFormatter(logging.Formatter):

    CYAN = '\033[96m'
    PINK = '\033[95m'
    PURPLE = '\033[94m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    GREY = '\033[90m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    format = "%(name)s - %(message)s"

    FORMATS = {
        logging.DEBUG: GREY + format + RESET,
        logging.INFO: RESET + format + RESET,
        logging.WARNING: YELLOW + format + RESET,
        logging.ERROR: RED + format + RESET,
        logging.CRITICAL: PINK + format + RESET
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# When register logger, use the formatter:
console = logging.StreamHandler()
console.setFormatter(CustomFormatter())
logger.addHandler(console)
```