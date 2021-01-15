import sys
import logging
import time
import os

def get_logging_handle(dir):

    log_format = '%(asctime)s %(message)s'

    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    file_name = 'log.txt'
    fh = logging.FileHandler(os.path.join(dir, file_name))
    fh.setFormatter(logging.Formatter(log_format))

    logging.getLogger().addHandler(fh)

    return logging
