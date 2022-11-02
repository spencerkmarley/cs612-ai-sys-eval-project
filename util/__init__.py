import os
from datetime import datetime as dt

from .pytorch_functions import *

from .model import (
    open_model,
    load_model,
    save_model,
    train,
    test,
)

from .nad import (
    NAD_train,
    NAD_test,
)

from .config import *

#
# Configure centralized logging facilities
#
import logging
import logging.config
import sys

# These are set in config.py - explicit call here so the code is more readable
LOG_DIR = LOG_DIR
BASE_LOG_FILENAME = BASE_LOG_FILENAME 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
file_formatter = logging.Formatter('%(asctime)s::%(levelname)s::%(message)s')

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)   # Set to DEBUG to see all messages
ch.setFormatter(console_formatter)
logger.addHandler(ch)


# File handler
# Time stamp for unique log suffix
now = dt.now()
log_filename = f'{BASE_LOG_FILENAME}_{now.strftime("%Y%m%d_%H%M%S")}.log'
log_filename = os.path.join(LOG_DIR, log_filename)

fh = logging.FileHandler(log_filename)
fh.setLevel(logging.DEBUG)
fh.setFormatter(file_formatter)
logger.addHandler(fh)

