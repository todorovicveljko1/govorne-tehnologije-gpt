import numpy as np
import torch


def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix]) # same as x but offset for 1

    return x,y