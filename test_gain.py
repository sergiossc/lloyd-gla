#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sergiossc@gmail.com
"""
import numpy as np
from utils import *
from numpy.linalg import svd


if __name__ == '__main__':
    print ('Precoding on perfect CSI\n') 
    n_tx = 4
    n_rx = 4
    
    l = 100 # number of training samples
    k = n_tx # codebook length
    
    dft_cb = gen_dftcodebook(k)
    for cw in dft_cb:
        print ('++++')
        x = np.array(cw)
        y = np.array(cw)
        
        z = np.kron(x,y)
        print (f'norm of z: {norm(z)}')
        print (f'abs of z: {np.abs(z)}')
        print (f'angle of z: {np.rad2deg(np.angle(z))}')
        
    
    variance = 1.0
    seed = 12345

    #dft_samples = gen_samples(dft_cb, l, variance, seed)
    non_dft_samples = gen_samples(None, l, variance, seed, n_rx, n_tx)

    samples = non_dft_samples

    #initial_codebook = np.array([samples[i] for i in np.random.choice(len(samples), k, replace=False)])
   
    for sample in samples:
        pass
#        print ('+++++')
#        h_mat = np.array(h_vec).reshape(n_rx, n_tx)
#        h_mat_normalized = h_mat/norm(h_mat)
#        print (f'norm mat: {norm(h_mat)}')
#        print (f'norm mat normalized: {norm(h_mat_normalized)}')
#        s, d, vh = svd(h_mat)
#        print (f's.shape: {s.shape}')
#        print (f'd.shape: {d.shape}')
#        print (f'vh.shape: {vh.shape}')
#        f = vh.conj().T
#        for i in f:
#            i = np.array(i).reshape(1, n_tx)
#            #print (i.shape)
#            pass
#
#
