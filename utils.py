import numpy as np


def cross_val(n = 1073, k = 10):
    for i in range(n//k):
        yield np.concatenate((np.arange(0,i*k, 1), np.arange((i+1)*k,n, 1))), np.arange(i*k,(i+1)*k, 1)
