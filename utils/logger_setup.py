import logging
import sys
import os
import time

'''
    get a logger depend on distributed_rank
'''
def setup_root_logger(save_dir, distributed_rank, filename="train_dhxy.log"):
    # if current process is not master process, we create a child logger for it,
    # and don't propagate this child logger's message to the root logger.
    # We don't create any handlers to this child logger, so that no message will be ouput from this process.
    if distributed_rank > 0:
        logger_not_root = logging.getLogger(name=f"{__name__}.{distributed_rank}")
        logger_not_root.propagate = False
        return logger_not_root

    # if current process is master process, we create a root logger for it,
    # and create handlers for the root logger.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

    if save_dir:
        filename = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime()) + "_" + filename
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)
    else:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        root_logger.addHandler(ch)

    return root_logger