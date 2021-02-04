from utils import *
import numpy as np
from numpy.linalg import svd



num_tx_list = [4, 8, 16, 32, 64, 128]
for num_tx in num_tx_list:
    #num_tx = 16
    num_rx = 1
    
    num_of_samples = 1000
    
    samples_seed = None
    
    samples = gen_samples(None, num_of_samples, 1.0, samples_seed, num_rx, num_tx) 
    
    dft_codebook = gen_dftcodebook(num_tx, 10)
       
    dft_codebook_dict = matrix2dict(dft_codebook)

 
    opt_gain_sum = 0
    equal_gain_sum = 0
    dft_gain_sum = 0

    for s in samples:
        #print (f'\n')
        #s = np.array(s).reshape(num_rx, num_tx)
        ##print (f'sample shape: {np.shape(s)}\n')
        f_opt = s.conj().T/norm(s)
        f_equal = (1/num_tx) * np.exp(1j * np.angle(f_opt))


        #print (f'f_equal.abs: {np.abs(f_equal)}') 
        #print (f'f_equal.angle: {np.angle(f_equal)}') 
        #print (f'f.abs: {np.abs(f)}') 
        #print (f'f.angle: {np.angle(f)}') 
        ##print (f'precoding: {norm(f)}\n')
        ##print (f'f:\n{f}')
        opt_gain = np.sum(s.conj() * f_opt)
        equal_gain = np.sum(s.conj() * f_equal)

        opt_gain = np.abs(opt_gain) ** 2
        equal_gain = np.abs(equal_gain) ** 2
        max_cw_id, dft_gain = gain_distortion(s, dft_codebook_dict)
 
        opt_gain_sum += opt_gain
        equal_gain_sum += equal_gain
        dft_gain_sum += dft_gain
        ##print (f'gain: {gain}')
        #u, d, vh = svd(s.T)
        #print (f'd = np.sum(sqrt({np.sum(np.sqrt(d))}))\n') 
        #n = squared_norm(s)
        #print (f'norm: {n}')
        #print (f'*****')
    opt_gain_mean = opt_gain_sum/num_of_samples
    equal_gain_mean = equal_gain_sum/num_of_samples
    dft_gain_mean = dft_gain_sum/num_of_samples
    
    print (f'num_tx, opt_gain_meam, equal_gain_mean, dft_gain_mean: {num_tx, opt_gain_mean, equal_gain_mean, dft_gain_mean}')





