import os
import numpy as np
import pandas as pd
import random
import torch
import logging
import time

def setup_logger(filename, name):
    """
    Sets up the logger for intermediate printing of results
    """

    now = time.localtime()
    s_time = "%02d%02d%02d%02d%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    name = s_time + name

    logger_fh = logging.FileHandler(filename, mode='w')
    logger_fm = logging.Formatter('%(message)s')
    logger_fh.setFormatter(logger_fm)

    logger = logging.getLogger(name)      # root logger
    logger.setLevel(logging.INFO)
    logger.addHandler(logger_fh)          # set the new handler
    logger.propagate = False
    return logger

def reset_seed(seed=1):
    """
    Resets the pseudo-random number generators
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_flags(FLAGS):
    """
    Saves the arguments used by the user
    """
    outfile = os.path.join(FLAGS.outroot, FLAGS.folder, FLAGS.drug, 'flags.cfg')
    with open(outfile, 'w') as f:
        for arg in vars(FLAGS):
            f.write('--%s=%s\n'%(arg, getattr(FLAGS, arg)))

def mkdir(directory):
    """
    Utility function for recursively creating directories
    """
    directories = directory.split("/")   

    folder = ""
    for d in directories:
        folder += d + '/'
        if not os.path.exists(folder):
            print('creating folder: %s'%folder)
            os.mkdir(folder)