import random
import numpy as np
import torch
import sys
import logging


logger = logging.getLogger(__name__)



def init_logger(filename=None):

    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO ,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
    logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
    return logger

def early_stopping(value, best, cur_step, max_step):

    stop_flag = False
    update_flag = False

    if value > best:
        cur_step = 0
        best = value
        update_flag = True
    else:
        cur_step += 1
        if cur_step > max_step:
            stop_flag = True

    return best, cur_step, stop_flag, update_flag

def set_randomseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
