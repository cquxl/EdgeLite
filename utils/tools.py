
import os
from loguru import logger
import sys


def setup_logger(log_name, save_dir):
    filename = '%s.log' % log_name
    save_file = os.path.join(save_dir, filename)
    if os.path.exists(save_file):
        with open(save_file, "w") as log_file:
            log_file.truncate()
    logger.remove()
    logger.add(save_file, rotation="10 MB", format="{time} {level} {message}", level="INFO")
    logger.add(sys.stdout, colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
                      "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.info('This is the %s log' % log_name)
    return logger