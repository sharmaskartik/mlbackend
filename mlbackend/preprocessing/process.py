import numpy as np
import scipy.signal as signal
from torch.utils import data

def time_delay_embedding(args):
    data = args['data']
    targets = args['targets']
    window_size = args['window_size']
    jump = args.get('jump', 1) 

    starts = np.arange(0, data.shape[2] - window_size, jump)

    new_data = []
    new_targets = []

    for i, sample in enumerate(data):
        for start in starts:
            if start + window_size > data.shape[2]:
                break

            new_data.append(data[i, :, start: start + window_size])
            new_targets.append(targets[i])

    new_data = np.array(new_data)
    new_targets  = np.array(new_targets)

    return new_data, new_targets

def decimate(args):
    data = args['data']
    downsample_factor = args['downsample_factor']
    axis = args.get('axis', -1)
    return signal.decimate(data, downsample_factor, axis = axis), args['targets']

