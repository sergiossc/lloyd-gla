from utils import *
import numpy as np
from numpy.linalg import svd

num_tx = 4
num_rx = 1

num_of_samples = 8000

#samples = np.array([richscatteringchnmtx(num_tx, num_rx)+np.ones((num_rx, num_tx), dtype=complex) for n in range(num_of_samples)])
##samples = np.array([richscatteringchnmtx(num_tx, num_rx) for n in range(num_of_samples)])
dftcb = gen_dftcodebook(num_tx)
samples = gen_samples(dftcb, num_of_samples, 1.0, True) 
#samples = [sample/norm(sample) for sample in samples]

num_samples, num_rows, num_cols = samples.shape
max_norm = -np.Inf
max_sample = np.zeros((num_rows, num_cols), dtype=complex)
for s in samples:
    s_norm = norm(s)
    if s_norm > max_norm:
        max_norm = s_norm
        max_sample = s
print ('max_norm: ', max_norm)
print ('max_sample: \n', max_sample)

for i in range(1, num_cols):
    max_distance = -np.Inf
    max_distance_sample = np.zeros((num_rows, num_cols), dtype=complex)
    for s in samples:
        s_distance = norm(s - max_sample)
        if s_distance > max_distance:
            max_distance = s_distance
            max_distance_sample = s
    print ('max_distance: ', max_distance)
    print ('max_distance_sample: \n', max_distance_sample)
    max_sample = max_distance_sample

mean = complex_average(samples)
de_meaned = np.array([sample - mean for sample in samples])

#num_samples, num_rows, num_cols = de_meaned.shape

S = np.zeros((num_cols, num_cols), dtype=complex)
for col1 in range(num_cols):
    for col2 in range(num_cols):
        x = np.sum(de_meaned[:,:,col1].conj() * de_meaned[:,:,col2])/(num_samples-1)
        S[col1, col2] = x
        if col1 == col2:
            pass
            #print (np.power(x, 2))
#print ("S:\n", S)
#print ("matrix_rank(S):\n", np.linalg.matrix_rank(S))

#for sample in samples:
#    print ("******************")
#    #print (np.log(1 + np.sum(sample.conj() * sample)))
#    print (np.sqrt(np.sum(sample.conj() * sample)))
#    print (sample.shape)
#    #sample = sample/norm(sample)
#    u, s, vh = svd(sample)
#    print (np.sum(s#))
#    #print ("vh:\n", vh)
