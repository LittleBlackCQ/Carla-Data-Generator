import yaml
import logging 
import datetime
import os

def config_from_yaml(config_file):
    if config_file == None:
        raise RuntimeError
    with open(config_file, 'r') as f:
        collector_config = yaml.safe_load(f)
    return collector_config

def create_logger(log_dir='log'):
    log_file = 'log_generator_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(log_dir, log_file)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger