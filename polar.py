import numpy as np
import matplotlib.pyplot as plt
from utils import *

def plot_polar(samples, only_samples=True):
    nrows, ncols = samples.shape
    print (nrows, ncols)
    #fig, axes = plt.subplots(nrows, ncols, subplot_kw=dict(polar=True))
    fig, axes = plt.subplots(nrows, ncols, subplot_kw=dict(polar=True))
    
    #axes[1].plot(np.pi, 1, 'ro')
    for col in range(ncols):
        for row in range(nrows):
            a = np.angle(samples[row,col])
            r = np.abs(samples[row,col])

            axes[row, col].plot(0, 1, 'wo')
            axes[row, col].plot(a, r, 'ro')
    #    angle = np.angle(samples[0,col])
    #    length = np.abs(samples[0,col]
    #    axes[col].plot(angle, length, 'ro')
    plt.show()

codebook = gen_dftcodebook(4)
cb_avg = complex_average(codebook)
#samples = gen_samples(codebook, 1000, 0.1)

#plot_polar(cb_avg)
plot_polar(codebook)
