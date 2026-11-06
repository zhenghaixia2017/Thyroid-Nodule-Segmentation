from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def get_gpu_allocated(gpus = []):

    if len(gpus) == 0:
        ret = torch.cuda.max_memory_allocated()
    else:
        ret = sum([torch.cuda.max_memory_allocated(device = gpu) for gpu in gpus])
    
    return ret
