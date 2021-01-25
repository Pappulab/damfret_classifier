import logging
from datetime import datetime


# Get the current timestamp
timestamp = int(datetime.utcnow().timestamp())
logging_filename = 'session-%d.log' % timestamp

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    filename=logging_filename,
                    level=logging.INFO)
