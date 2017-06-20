import numpy as np

def dice_ratio(preds, labels):
    '''
    preds & labels should only contain 0 or 1.
    '''
    return np.sum(preds[labels==1])*2.0 / (np.sum(preds) + np.sum(labels))