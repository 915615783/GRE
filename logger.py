import logging

logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('./log/test.log', mode='w')
ch = logging.StreamHandler()

formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')

fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)