import os
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.metrics import mean_squared_error
import numpy as np


def PrintArgs(logger, args):
    message = ''
    for k, v in sorted(vars(args).items()):
        message += '\n{:>30}: {:<30}'.format(str(k), str(v))
    logger.info(message)
        

def SaveModel(model, optimizer, scheduler, args, epoch, savedir:str):
    ckpt = {
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args'                : args,
        'epoch'               : epoch
    }
    try:
        torch.save(ckpt, os.path.join(savedir, f'epoch_{str(epoch)}.pth'))
        torch.save(ckpt, os.path.join(savedir, 'latest.pth'))
    except:
        raise RuntimeError('Failed to save model!')