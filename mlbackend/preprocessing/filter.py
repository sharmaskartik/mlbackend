from scipy.signal import butter, filtfilt
import numpy as np

def get_butter_filter(freqs, fs, btype, order=5):
    nyq = 0.5 * fs
    if btype == 'band':
        low = freqs[0] / nyq
        high = freqs[1] / nyq
        b, a = butter(order, [low, high], btype='band')
    else:
        freq = freqs[0] / nyq
        b, a = butter(order, freq, btype= btype)
    return b, a


def butter_filter(args):
    data = args['data']
    freqs = args['freqs']
    fs = args['fs']
    order= args.get('order', 5)
    btype = args.get('btype', 'lp')

    if btype != 'lp' and btype != 'hp' and btype != 'band':
        raise Exception('Incorrect value passed for btype - $s. Should be either of lp, hp or band'%(btype))    
    
    nfreqs = np.array(freqs).shape[0]

    if (nfreqs <1 and nfreqs > 2) or (btype in ['lp', 'hp'] and nfreqs!=1) or (btype == 'band' and nfreqs!=2):
        raise Exception('Incorrect number of frequencies provided. Should be 1 for lp and hp. 2 for band')    

    b, a = get_butter_filter(freqs, fs, btype, order=order)
    y = filtfilt(b, a, data)
    return y, args['targets']


