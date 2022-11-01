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


#
# Configure centralized logging facilities
#
import logging
import logging.config
import sys

# logging.basicConfig(
#     format='%(asctime)s::%(levelname)s::%(message)s',
#     level = logging.DEBUG,
#     handlers=[
#         logging.FileHandler('logs/bd_detection.log'),
#         logging.StreamHandler(sys.stdout)
#     ],
# )

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
file_formatter = logging.Formatter('%(asctime)s::%(levelname)s::%(message)s')

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)   # Set to DEBUG to see all messages
ch.setFormatter(console_formatter)
logger.addHandler(ch)

fh = logging.FileHandler('logs/bd_detection.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(file_formatter)
logger.addHandler(fh)

