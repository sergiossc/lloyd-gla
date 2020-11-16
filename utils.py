#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sergiossc@gmail.com
"""
import numpy as np
import uuid
import matplotlib.pyplot as plt
import json
from scipy.linalg import hadamard

def squared_norm(cw):
    """
    Input: cw as a vector (1-dim)
    Output: return a squared norm as a inner product of cw.conj() * cw
    """
    inner_product = np.sum(cw.conj() * cw)
    return inner_product

def norm(cw): 
    return np.sqrt(squared_norm(cw))


def gen_dftcodebook(num_of_cw):
    tx_array = np.arange(num_of_cw)
    mat = np.matrix(tx_array).T * tx_array
    cb = np.exp(1j * 2 * np.pi * mat/num_of_cw)
    return cb
    
def richscatteringchnmtx(num_tx, num_rx, variance):
    """
    Ergodic channel. Fast, frequence non-selective channel: y_n = H_n x_n + z_n.  
    Narrowband, MIMO channel
    PDF model: Rich Scattering
    Circurly Simmetric Complex Gaussian from: 
         https://www.researchgate.net/post/How_can_I_generate_circularly_symmetric_complex_gaussian_CSCG_noise
    """
    sigma = variance
    #my_seed = 2323
    #np.random.seed(my_seed)
    h = np.sqrt(sigma/2)*(np.random.randn(num_rx, num_tx) + np.random.randn(num_rx, num_tx) * 1j)
    #h = np.sqrt(sigma/2)*np.random.randn(num_tx, num_rx)
    return h

def gen_samples(codebook, num_of_samples, variance, same_samples_for_all=True):
    if same_samples_for_all:
        samples_seed = 789
        np.random.seed(samples_seed)
    num_rows = np.shape(codebook)[0]
    num_cols = np.shape(codebook)[1]
    samples = []
    for n in range(int(num_of_samples/num_rows)):
        for cw in codebook:
            noise = np.sqrt(variance/(2*num_cols)) * (np.random.randn(1, num_cols) + np.random.randn(1, num_cols) * 1j)
            sample = cw + noise
            samples.append(sample)
    np.random.seed(None)
    return np.array(samples)

def complex_average(samples):
    return np.mean(samples, axis=0)

def duplicate_codebook(codebook, perturbation_vector):
    new_codebook = []
    for cw in codebook:
        cw1 = cw + perturbation_vector
        cw2 = cw - perturbation_vector
        new_codebook.append(cw1)
        new_codebook.append(cw2)
    return np.array(new_codebook)

def dict2matrix(dict_info):
    vector = []
    for k, v in dict_info.items():
        vector.append(v)
    return np.array(vector)

def matrix2dict(matrix):
    dict_info = {}
    for l in matrix:
        id = uuid.uuid4()
        dict_info[id] = l
    return dict_info

def sorted_samples(samples, attr='norm'):
    nsamples, nrows, ncols = samples.shape
    s_not_sorted = []

    if attr == 'norm': #Sorted by vector norm   ??????
        for s in samples:
            s_norm = np.abs(norm(s))
            s_info = {}
            s_info = {'s_norm': s_norm, 's': s}
            s_not_sorted.append(s_info)

        s_sorted = sorted(s_not_sorted, key=lambda k: k['s_norm'])
        samples_sorted = [v['s'] for v in s_sorted]
        attr_sorted = [v['s_norm'] for v in s_sorted]

    elif attr == 'mse': #Sorted by vector norm   ??????
        s_avg = complex_average(samples)
        for s in samples:
            s_mse = norm(s-s_avg)
            s_info = {}
            s_info = {'s_mse': s_mse, 's': s}
            s_not_sorted.append(s_info)

        s_sorted = sorted(s_not_sorted, key=lambda k: k['s_mse'])
        samples_sorted = [v['s'] for v in s_sorted]
        attr_sorted = [v['s_mse'] for v in s_sorted]



    elif attr == 'stddev':  #Sorted by Standard Deviation

        s_avg = complex_average(samples)
        for s in samples:
            s_de_meaned = s - s_avg
            s_stddev = squared_norm(s_de_meaned)/ncols
            s_info = {}
            s_info = {'s_stddev': s_stddev, 's': s}
            s_not_sorted.append(s_info)

        s_sorted = sorted(s_not_sorted, key=lambda k: k['s_stddev'])
        samples_sorted = [v['s'] for v in s_sorted]
        attr_sorted = [v['s_stddev'] for v in s_sorted]
        var = sum(attr_sorted)/len(attr_sorted)
        std = np.sqrt(var)
        print ("var: ", var)
        print ("std: ", std)

    elif attr == 'abs_mean_characteristic_value': # From the paper

        for s in samples:
            num_rx, num_tx = s.shape
            #print (num_rx, num_tx)
            #print ("s:\n", s)
            s_mean = np.sum(s)/num_tx
            #print ("s_avg:\n", s_avg)
            s_info = {}
            s_info = {'s_abs_mean': np.abs(s_mean), 's': s}
            s_not_sorted.append(s_info)

        s_sorted = np.array(sorted(s_not_sorted, key=lambda k: k['s_abs_mean']))
        samples_sorted = [v['s'] for v in s_sorted]
        attr_sorted = [v['s_abs_mean'] for v in s_sorted]

    elif attr == 'variance_characteristic_value': # From the paper

        #s_avg = complex_average(samples)
        for s in samples:
            num_rx, num_tx = s.shape
            s_avg = np.sum(s)/num_tx
            s_demeaned = s - s_avg
            s_var = np.sqrt(np.sum(s_demeaned.conj() * s_demeaned)/num_tx)
            s_info = {}
            s_info = {'s_var': np.abs(s_var), 's': s}
            s_not_sorted.append(s_info)

        s_sorted = np.array(sorted(s_not_sorted, key=lambda k: k['s_var']))
        samples_sorted = [v['s'] for v in s_sorted]
        attr_sorted = [v['s_var'] for v in s_sorted]


    else:
        return -1

    return np.array(samples_sorted), np.array(attr_sorted)


def mse_distortion(sample, codebook_dict):
    min_mse = np.Inf
    min_cw_id = None
    for cw_id, cw in codebook_dict.items():
        mse = squared_norm(cw - sample)/np.size(sample)
        if mse < min_mse:
            min_mse = mse
            min_cw_id = cw_id
    return min_cw_id, min_mse

def gain_distortion(sample, codebook_dict):
    """
    Input: sample: a complex unitary vector with norm(sample) == 1;
           codebook_dict: a dict of codewords. Each codeword is a unitary complex vector with norm(codeword) == 1. 
    Output: codeword_id: uuid from codeword who produces the max_gain
            max_gain: real-valued gain from cosine formula.
    """
    max_gain = -np.Inf
    max_cw_id = None
    for cw_id, cw in codebook_dict.items():
        gain = np.abs(np.inner(sample.conj(), cw))
        if gain > max_gain:
            max_gain = gain
            max_cw_id = cw_id
    return max_cw_id, max_gain


def xiaoxiao_initial_codebook(samples):

    num_samples, num_rows, num_cols = samples.shape

    # Code samples in hadamard code
    samples_hadamard = hadamard_transform(samples, False)    

    # Ordering samples by variance characteristic value (ascending way)
    samples_sorted, attr_sorted = sorted_samples(samples_hadamard, 'variance_characteristic_value') 
    
    # Index A, B and C groups
    a_group_begin = 0
    a_group_end = 17 * int(num_samples/20)

    b_group_begin = a_group_end
    b_group_end = b_group_begin + (2 * int(num_samples/20))

    c_group_begin = b_group_end
    c_group_end = -1 

    # Getting samples from ordered samples spliting in groups as indexed as before
    a_group_of_samples = samples_sorted[a_group_begin:a_group_end, :, :]
    b_group_of_samples = samples_sorted[b_group_begin:b_group_end, :, :]
    c_group_of_samples = samples_sorted[c_group_begin:c_group_end, :, :]
    
    # Ordering subgroups by mean characteristic value
    samples_a_group_sorted, attr_a_group_sorted = sorted_samples(a_group_of_samples, 'abs_mean_characteristic_value') 
    samples_b_group_sorted, attr_a_group_sorted = sorted_samples(b_group_of_samples, 'abs_mean_characteristic_value') 
    samples_c_group_sorted, attr_a_group_sorted = sorted_samples(c_group_of_samples, 'abs_mean_characteristic_value') 

    # For each subgroup, select the codewords. Ex.: all/2, all/4 and all/4 number of codewords
    num_of_codewords = num_cols

    #print ('len(group_a): ', len(samples_a_group_sorted))
    index_a = get_index_codewords_from_sub_samples(len(samples_a_group_sorted), num_of_codewords/2)
    #print ('index_a:', index_a)

    #print ('len(group_b): ', len(samples_b_group_sorted))
    index_b = get_index_codewords_from_sub_samples(len(samples_b_group_sorted), num_of_codewords/4)
    #print ('index_b:', index_b)

    #print ('len(group_c): ', len(samples_c_group_sorted))
    index_c = get_index_codewords_from_sub_samples(len(samples_c_group_sorted), num_of_codewords/4)
    #print ('index_c:', index_c)


    #igetting codewords from subgroups
    list_initial_codebook_from_a_group = [samples_a_group_sorted[i] for i in index_a]
    list_initial_codebook_from_b_group = [samples_b_group_sorted[i] for i in index_b]
    list_initial_codebook_from_c_group = [samples_c_group_sorted[i] for i in index_c]

    initial_codebook = np.array(list_initial_codebook_from_a_group + list_initial_codebook_from_b_group + list_initial_codebook_from_c_group)

    #print (initial_codebook.shape)
    return initial_codebook, samples_hadamard

def get_index_codewords_from_sub_samples(n_samples, n_codewords):

    slot = int(n_samples/n_codewords)
    step = slot/2

    index_codebook_list = []

    for n in range(int(n_codewords)):
            start = n * slot
            mid = start + step
            index_codebook_list.append(int(mid))

    return index_codebook_list



def katsavounidis_initial_codebook(samples):

    num_samples, num_rows, num_cols = samples.shape


    samples_dict = matrix2dict(samples)
        
    max_norm = -np.Inf
    max_sample_id = ''

    for s_id, s in samples_dict.items():
        s_norm = norm(s)
        if s_norm > max_norm:
            max_norm = s_norm
            max_sample_id = s_id
    
    num_of_codewords = num_cols
    initial_codebook = np.zeros((num_of_codewords, num_rows, num_cols), dtype=complex)
    
    # Remove the max_sample_id from samples_dict and add it as our first codeword in initial_codebook
    initial_codebook[0,:,:] = samples_dict.pop(max_sample_id) 

    # Step 2: Define 2nd codeword as the largest distance from the 1st codeword
    cw = initial_codebook[0,:,:]
    max_distance = -np.Inf
    max_distance_sample_id = '' 
    for s_id, s in samples_dict.items():
        s_distance = norm(s - cw)
        if s_distance > max_distance:
            max_distance = s_distance
            max_distance_sample_id = s_id
    initial_codebook[1,:,:] = samples_dict.pop(max_distance_sample_id)

    # Step 3: defining next codewords

    for i in range(0, num_of_codewords - 2):

        min_distance = np.Inf
        min_distance_sample_id = '' 

        for s_id, s in samples_dict.items():
            s_distance = 0
            for codeword in initial_codebook:
                s_distance = s_distance + norm(s - codeword)
            if s_distance < min_distance:
                min_distance = s_distance
                min_distance_sample_id = s_id
    
        #for s_id, s in samples_dict.items():
        #    for codeword in initial_codebook:
        #        s_distance = norm(s - codeword)
        #        if s_distance < min_distance:
        #            min_distance = s_distance
        #            min_distance_sample_id = s_id
    
        max_distance = -np.Inf
        max_distance_sample_id = '' 

        for s_id, s in samples_dict.items():
    
            s_distance = norm(s - samples_dict[min_distance_sample_id])
            if s_distance > max_distance:
                max_distance = s_distance
                max_distance_sample_id = s_id
    
    
        initial_codebook[i+2,:,:] = samples_dict.pop(max_distance_sample_id)

    return initial_codebook
 

def perform_distortion(sample, codebook_dict, metric):
    cw_id = None
    distortion = None
    distortion_opts = {'mse': mse_distortion, 'gain': gain_distortion}
    distortion_function = distortion_opts.get(metric, None)
    cw_id, distortion = distortion_function(sample, codebook_dict)
    return cw_id, np.abs(distortion)

def lloyd_gla(initial_alphabet_opt, samples, num_of_levels, num_of_iteractions, distortion_measure, perturbation_variance=None, initial_codebook=None, percentage_of_sub_samples=None):
    """
        This method implements Lloyd algorithm. There are two options of initial reconstruct alphabet: (1) begining a unitary codebook and duplicate it in each round. The number of rounds is log2(num_of_levels). And (2) randomized initial reconstruct alphabet from samples.
    """
    if initial_alphabet_opt == 'unitary_until_num_of_elements':
        cw0 = complex_average(samples) # The inicial unitary codebook is a average of all samples
        #plot_unitary_codebook(cw0, 'initial_codebook.png')
        cw0_shape = np.shape(cw0)
        codebook = []    
        codebook.append(cw0)
        codebook = np.array(codebook)

        #This method considers a perturbation vector to duplicate the codebook on each round
        perturbation_vector = np.sqrt(perturbation_variance/2) * (np.random.randn(cw0_shape[0], cw0_shape[1]) + 1j * np.random.randn(cw0_shape[0], cw0_shape[1]))
        #perturbation_vector = np.sqrt(0.001/2) * (np.random.randn(cw0_shape[0], cw0_shape[1]) + 1j * np.random.randn(cw0_shape[0], cw0_shape[1]))
        #print ('perturbation_vector:\n')
        #print (perturbation_vector)

        num_of_rounds = int(np.log2(num_of_levels))

        #nsamples, nrows, ncols = samples_sorted.shape
        #p = int(nsamples/num_of_rounds)
        #samples_start = 0
        #samples_end = p
        

    elif initial_alphabet_opt == 'random_from_samples':
        initial_codebook_from_samples = [samples[i] for i in np.random.choice(len(samples), num_of_levels, replace=False)]
        codebook = np.array(initial_codebook_from_samples)
        num_of_rounds = 1 # for randomized initial alphabet method only one round is needed

    elif initial_alphabet_opt == 'sa':
        if initial_codebook.all() == None:
            codebook = np.array([samples[i] for i in np.random.choice(len(samples), num_of_levels, replace=False)])
        else:
            codebook = initial_codebook
        num_of_rounds = 1 # for randomized initial alphabet method only one round is needed
       
    elif initial_alphabet_opt == 'user_defined':
        codebook = initial_codebook
        num_of_rounds = 1 # for initial alphabet from user method only one round is needed
 


    elif initial_alphabet_opt == 'from_sorted_samples':
        #samples_sorted, attr_sorted = sorted_samples(samples, 'stddev')
        samples_sorted, attr_sorted = sorted_samples(samples, 'mse')
        samples = samples_sorted
        num_of_rounds = 1

        nsamples, nrows, ncols = samples.shape
        p = int(nsamples/num_of_levels)

        index_codebook_list = []

        for n_level in range(num_of_levels):
            start = n_level * p
            end = start + p
            index_codebook_list.append(int((start+end)/2))

        codebook = np.array([samples[i] for i in index_codebook_list])
 
 
    else:
        return None


    mean_distortion_by_round = {}
    current_codebook_dict = None
    mean_distortion_by_iteractions = None


    for r in range(1, num_of_rounds+1):
        if initial_alphabet_opt == 'unitary_until_num_of_elements':
            codebook = duplicate_codebook(codebook, perturbation_vector)
            #plot_codebook(codebook, 'duplicated_codebook_from_round'+str(r)+'.png')
            #samples = samples_sorted[samples_start:samples_end]
            #samples_start = samples_start + p 
            #samples_end = samples_end + p
        elif initial_alphabet_opt == 'random_from_samples':
            pass
        elif initial_alphabet_opt == 'sa':
            pass
        elif initial_alphabet_opt == 'user_defined':
            pass
        elif initial_alphabet_opt == 'from_sorted_samples':
            pass
        else:
            return None

        samples_dict = matrix2dict(samples)
        mean_distortion_by_iteractions = [] #np.zeros(num_of_iteractions)

        for n in range(num_of_iteractions):
            codebook_dict = matrix2dict(codebook)

            sets = {}  # Storage information of partitions baised by each codewords
            for cw_id in codebook_dict.keys():
                sets[cw_id] = []

            distortion = 0  # Distortion measurement of this interaction
            for sample_id, sample in samples_dict.items():
                cw_id, estimated_distortion = perform_distortion(sample, codebook_dict, distortion_measure)
                distortion = distortion + estimated_distortion
                sample_info = {'sample_id': sample_id, 'est_distortion': estimated_distortion}
                #sets[cw_id].append(sample_id)
                sets[cw_id].append(sample_info)
            mean_distortion = distortion/len(samples) 
            mean_distortion_by_iteractions.append(mean_distortion)
            if (n>0) and (mean_distortion_by_iteractions[n-1] == mean_distortion_by_iteractions[n]):
                break
 
            current_codebook_dict = codebook_dict.copy()            
            # May I could get out here from interactions with a designed codebook if mean distortion variation is acceptable. Otherwise, I have to design a new codebook from sets.

            # Designing a new codebook from sets
            new_codebook_dict = {}
            for cw_id, samples_info_list in sets.items():
                if len(samples_info_list) > 0:
                    samples_sorted = sorted(samples_info_list, key=lambda k: k['est_distortion'])
                    #print ([sample_info['est_distortion'] for sample_info in samples_sorted])
                    sub_set_of_samples = {}
                    for sample_info in samples_sorted:
                        sample_id = sample_info['sample_id']
                        sub_set_of_samples[sample_id] = samples_dict[sample_id]
                    if len(sub_set_of_samples) > 2:
                        sub_set_of_samples_matrix = dict2matrix(sub_set_of_samples) 
                        start = 0
                        if percentage_of_sub_samples is None:
                            percentage_of_sub_samples = 1 # Ex.: 0.8 is 80% of subsamples
                        end = int(len(sub_set_of_samples) * percentage_of_sub_samples)
                        new_cw = complex_average(sub_set_of_samples_matrix[start:end])
                    else:
                        new_cw = complex_average(dict2matrix(sub_set_of_samples))
      
                    new_cw = new_cw/norm(new_cw)
                else:
                    if initial_alphabet_opt == 'random_from_samples' or initial_alphabet_opt == 'sa' or initial_alphabet_opt == 'user_defined':
                        #new_cw = codebook_dict[cw_id] # Enable this line to keep the cw who has 0 samples, but for a better design it should be removed from codebook.
                        new_cw_index = np.random.choice(len(samples))
                        new_cw = np.array(samples[new_cw_index]) # this is more interesting: if cw had groupped any sample, get another one from samples.
                    elif initial_alphabet_opt == 'unitary_until_num_of_elements':
                        #new_cw_index = np.random.choice(len(samples))
                        #new_cw = np.array(samples[new_cw_index]) # this is more interesting: if cw had groupped any sample, get another one from samples.
                        #new_cw = codebook_dict[cw_id] # In this case, keep the same codeword. May there are a better solution... 
                        new_cw = np.array(cw0)
                        #pass

                new_codebook_dict[cw_id] = new_cw
            codebook = dict2matrix(new_codebook_dict)
        #plot_codebook(codebook, 'designed_codebook_from_round'+str(r)+'.png')
        mean_distortion_by_round[r] = mean_distortion_by_iteractions

    return dict2matrix(current_codebook_dict), sets,  mean_distortion_by_round

# Some plot functions

def plot_unitary_codebook(codebook, filename):
    nrows, ncols = codebook.shape
    fig, axes = plt.subplots(nrows, ncols, subplot_kw=dict(polar=True))
    for col in range(ncols):
        for row in range(nrows):
            a = np.angle(codebook[row,col])
            r = np.abs(codebook[row,col])
            if nrows == 1:
                axes[col].plot(0, 1, 'wo')
                axes[col].plot(a, r, 'ro')
            else:
                axes[row, col].plot(0, 1, 'wo')
                axes[row, col].plot(a, r, 'ro')
    plt.savefig(filename)

def plot_codebook(codebook, filename):
    ncodewords, nrows, ncols = codebook.shape
    #nrows, ncols = codebook.shape
    fig, axes = plt.subplots(ncodewords, ncols, subplot_kw=dict(polar=True))
    #fig, axes = plt.subplots(1, ncols, subplot_kw=dict(polar=True))
    #print (axes.shape)
    for col in range(ncols):
        for cw in range(ncodewords):
            a = np.angle(codebook[cw, 0, col])
            r = np.abs(codebook[cw, 0, col])
            axes[cw, col].plot(0, 1, 'wo')
            axes[cw, col].plot(a, r, 'ro')
    plt.savefig(filename)

def plot_polar_samples(samples, filename):
    nsamples, nrows, ncols = samples.shape
    fig, axes = plt.subplots(nrows, ncols, subplot_kw=dict(polar=True))

    for n in range(nsamples):
        for col in range(ncols):
            a = np.angle(samples[n, 0, col])
            r = np.abs(samples[n, 0, col])
            axes[col].plot(a, r, 'o')
    plt.savefig(filename)

def plot_samples(samples, filename, title, y_label):
    fig, ax = plt.subplots()
    #print (samples)
    #nsamples, nrows, ncols = samples.shape
    #x = np.arange(nsamples)
    #y = samples
    ax.scatter(x=np.arange(len(samples)), y=np.abs(samples), marker='o', c='r', edgecolor='b')
    ax.set_xlabel('samples')
    ax.set_ylabel(y_label)
    plt.title(title)
    plt.savefig(filename)



# JSON STUFF TO ENCODE/DECODE DATA
def encode_codebook(codebook):
    codebook_enc = {}
    for cw_id, cw in codebook.items():
        adjust = {}
        count = 0
        codeword = np.array(cw).reshape(cw.size)
        for complex_adjust in codeword:
            adjust_id = str('complex_adjust') + str(count)
            adjust[adjust_id] = (complex_adjust.real, complex_adjust.imag)
            count += 1
        codebook_enc[str(cw_id)] = adjust
    return codebook_enc

def encode_sets(sets):
    sets_enc = {}
    for cw_id, samples_id_list in sets.items():
        sets_enc[str(cw_id)] = len(samples_id_list)
    return sets_enc

def encode_mean_distortion(distortion_by_round):
    distortion_enc = {}
    for r, distortion_by_interactions in distortion_by_round.items():
        count = 0
        distortion_by_interactions_enc = {}
        for d in distortion_by_interactions:
            distortion_by_interactions_enc[str(count)] = float(d)
            count += 1
        distortion_enc[str(r)] = distortion_by_interactions_enc
    return distortion_enc

def decode_codebook(codebook_json):
    codebook_dict = {}
    for cw_id, cw in codebook_json.items():
        codeword = []
        for cw_adjust in cw.items():
            real_adjust = cw_adjust[1][0]
            imag_adjust = cw_adjust[1][1]
            adjust = real_adjust + 1j * imag_adjust
            codeword.append(adjust)
        codebook_dict[cw_id] = np.array(codeword, dtype=complex)
    return codebook_dict

#def save_training_samples(samples):
#    np.save('samples.npy', samples)

#def load_samples(filename):
#    samples = np.load(filename)
#    return samples


#def std_deviation(vector):
#    de_meaned = vector - average(vector)
#    return norm(de_meaned) * 1/np.sqrt(len(vector))

#def rms():
#    return np.sqrt(np.power(average(x), 2) + np.power(std_deviation(x), 2))

#def complex_correlation(cw1, cw2):
#    cw1 = cw1/norm(cw1)
#    cw2 = cw2/norm(cw2)
#    u = np.matrix([np.real(cw1), np.imag(cw1)])
#    u_vec = np.array(u).reshape(np.size(u))
#    v = np.matrix([np.real(cw2), np.imag(cw2)])
#    v_vec = np.array(v).reshape(np.size(v))
#    correlation = np.inner(u_vec, v_vec)
#    return correlation

#def correlation_factor(x, y):
#    de_meaned_x = x - average(x)
#    de_meaned_y = y - average(y)
#    return np.inner(de_meaned_x, de_meaned_y) / (norm(de_meaned_x) * norm(de_meaned_y))

#def get_mean_distortion(sets, samples, codebook):
#    sum_squared_error = 0
#    for cw_id, samples_id_list in sets.items():
#        cw = codebook[cw_id]
#        for sample_id in samples_id_list:
#            sample = samples[sample_id]    
#            squared_error = np.sum(complex_squared_error(cw, sample))
#            sum_squared_error += squared_error
#    return sum_squared_error/len(samples)

def plot_performance(distortion_by_round, graph_title, filename):
    fig, ax = plt.subplots()
    for r, mean_distortion in distortion_by_round.items():
        ax.plot(mean_distortion, label='#cw: ' + str(2**r))
    plt.ylabel('distortion (MSE)')
    plt.xlabel('# iterations')
    plt.title(graph_title)
    plt.legend()
    fig.savefig(filename)

def hadamard_transform(samples, inverse=False):
    num_samples, num_rows, num_cols = samples.shape
    hadamard_mat = hadamard(int(num_cols), dtype=complex)
    samples_converted = []

    for s in samples:
        s = s.reshape(num_cols)
        s_h = np.zeros((num_cols), dtype=complex)
        for n in range(num_cols):
            s_h[n] = np.sum(hadamard_mat[n].conj() * s)
        if inverse:
            s_h = np.array(s_h).reshape(1, num_cols) * (1/num_cols)
        else:
            s_h = np.array(s_h).reshape(1, num_cols) 
        
        samples_converted.append(s_h)
    samples_converted = np.array(samples_converted)

    return samples_converted
  


