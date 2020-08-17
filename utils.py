#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sergiossc@gmail.com
"""
import numpy as np
import uuid
import matplotlib.pyplot as plt
import json

def squared_norm(cw):
    return np.sum(np.power(np.abs(cw), 2))

def norm(cw):
    return np.sqrt(squared_norm(cw))

def std_deviation(vector):
    de_meaned = vector - average(vector)
    return norm(de_meaned) * 1/np.sqrt(len(vector))

def rms():
    return np.sqrt(np.power(average(x), 2) + np.power(std_deviation(x), 2))

def complex_correlation(cw1, cw2):
    cw1 = cw1/norm(cw1)
    cw2 = cw2/norm(cw2)
    u = np.matrix([np.real(cw1), np.imag(cw1)])
    u_vec = np.array(u).reshape(np.size(u))
    v = np.matrix([np.real(cw2), np.imag(cw2)])
    v_vec = np.array(v).reshape(np.size(v))
    correlation = np.inner(u_vec, v_vec)
    return correlation

def correlation_factor(x, y):
    de_meaned_x = x - average(x)
    de_meaned_y = y - average(y)
    return np.inner(de_meaned_x, de_meaned_y) / (norm(de_meaned_x) * norm(de_meaned_y))

def gen_dftcodebook(num_of_cw):
    tx_array = np.arange(num_of_cw)
    mat = np.matrix(tx_array).T * tx_array
    cb = np.exp(1j * 2 * np.pi * mat/num_of_cw)
    return cb
    
def gen_samples(codebook, num_of_samples, variance, same_samples_for_all=True):

    if same_samples_for_all:
        samples_seed = 789
        np.random.seed(samples_seed)

    num_rows = np.shape(codebook)[0]
    num_coluns = np.shape(codebook)[1]
    samples = []
    for n in range(int(num_of_samples/num_rows)):
        for cw in codebook:
            noise = np.sqrt(variance/2) * (np.random.randn(1, num_coluns) + np.random.randn(1, num_coluns) * 1j)
            sample = cw + noise
            samples.append(sample)
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

def get_mean_distortion(sets, samples, codebook):
    sum_squared_error = 0
    for cw_id, samples_id_list in sets.items():
        cw = codebook[cw_id]
        for sample_id in samples_id_list:
            sample = samples[sample_id]    
            squared_error = np.sum(complex_squared_error(cw, sample))
            sum_squared_error += squared_error
    return sum_squared_error/len(samples)

def plot_performance(distortion_by_round, graph_title, filename):
    fig, ax = plt.subplots()
    for r, mean_distortion in distortion_by_round.items():
        ax.plot(mean_distortion, label='#cw: ' + str(2**r))
    plt.ylabel('distortion (MSE)')
    plt.xlabel('# iterations')
    plt.title(graph_title)
    plt.legend()
    fig.savefig(filename)

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

def mse_distortion(sample, codebook_dict):
    min_mse = np.Inf
    min_cw_id = None
    for cw_id, cw in codebook_dict.items():
        mse = squared_norm(cw - sample) 
        if mse < min_mse:
            min_mse = mse
            min_cw_id = cw_id
    return min_cw_id, min_mse

def gain_distortion(sample, codebook_dict):
    max_gain = -np.Inf
    max_cw_id = None
    for cw_id, cw in codebook_dict.items():
        #gain = np.abs(np.inner(sample.conj(), cw))
        gain = np.abs(np.inner(sample, cw))
        if gain > max_gain:
            max_gain = gain
            max_cw_id = cw_id
    return max_cw_id, max_gain
 

def perform_distortion(sample, codebook_dict, metric):

    cw_id = None
    distortion = None

    distortion_opts = {'mse': mse_distortion, 'gain': gain_distortion}
    distortion_function = distortion_opts.get(metric, None)

    cw_id, distortion = distortion_function(sample, codebook_dict)

    return cw_id, distortion


def lloyd_lbg(initial_alphabet_opt, samples, num_of_levels, num_of_iteractions, distortion_measure, perturbation_variance=None):
    """
        This method implements Lloyd algorithm. There are two options of initial reconstruct alphabet: (1) begining a unitary codebook and duplicate it in each round. The number of rounds is log2(num_of_levels). And (2) randomized initial reconstruct alphabet from samples.
    """
    if initial_alphabet_opt == 'unitary_until_num_of_elements':
        cw0 = complex_average(samples) # The inicial unitary codebook is a average of all samples
        cw0_shape = np.shape(cw0)
        codebook = []    
        codebook.append(cw0)
        codebook = np.array(codebook)

        #This method considers a perturbation vector to duplicate the codebook on each round
        perturbation_vector = np.sqrt(perturbation_variance/2) * (np.random.randn(cw0_shape[0], cw0_shape[1]) + 1j * np.random.randn(cw0_shape[0], cw0_shape[1]))
        num_of_rounds = int(np.log2(num_of_levels))

    elif initial_alphabet_opt == 'random_from_samples':
        initial_codebook_from_samples = [samples[i] for i in np.random.randint(0, len(samples), num_of_levels)]
        codebook = np.array(initial_codebook_from_samples)
        num_of_rounds = 1 # for randomized initial alphabet method only one round is necessary

    else:
        return None


    mean_distortion_by_round = {}
    current_codebook_dict = None
    mean_distortion_by_iteractions = None


    for r in range(1, num_of_rounds+1):


        if initial_alphabet_opt == 'unitary_until_num_of_elements':
            codebook = duplicate_codebook(codebook, perturbation_vector)
        elif initial_alphabet_opt == 'random_from_samples':
            pass
        else:
            return None

        samples_dict = matrix2dict(samples)
        mean_distortion_by_iteractions = np.zeros(num_of_iteractions)

        for n in range(num_of_iteractions):

            codebook_dict = matrix2dict(codebook)
            
            sets = {}  # Storage information of partitions baised by each codewords
            for cw_id in codebook_dict.keys():
                sets[cw_id] = []

            distortion = 0  # Distortion measurement of this interaction
            for sample_id, sample in samples_dict.items():
                cw_id, estimated_distortion = perform_distortion(sample, codebook_dict, distortion_measure)
                distortion = distortion + estimated_distortion
                sets[cw_id].append(sample_id)
            mean_distortion_by_iteractions[n] = distortion/len(2*samples) 
            current_codebook_dict = codebook_dict.copy()            
            # May I could get out here from interactions with a designed codebook if mean distortion variation is acceptable. Otherwise, I have to design a new codebook from sets.

            # Designing a new codebook from sets
            new_codebook_dict = {}
            for cw_id, samples_id_list in sets.items():
                if len(samples_id_list) > 0:
                    sub_set_of_samples = {}
                    for sample_id in samples_id_list:
                        sub_set_of_samples[sample_id] = samples_dict[sample_id]
                    new_cw = complex_average(dict2matrix(sub_set_of_samples))
                    new_cw = new_cw/norm(new_cw)
                else:
                    if initial_alphabet_opt == 'random_from_samples':
                        #new_cw = codebook_dict[cw_id] # Enable this line to keep the cw who has 0 samples, but for a better design it should be removed from codebook.
                        new_cw_index = np.random.randint(0, len(samples))
                        new_cw = np.array(samples[new_cw_index]) # this is more interesting: if cw had groupped any sample, get another one from samples.

                new_codebook_dict[cw_id] = new_cw
            codebook = dict2matrix(new_codebook_dict)
        mean_distortion_by_round[r] = mean_distortion_by_iteractions

    return dict2matrix(current_codebook_dict), sets,  mean_distortion_by_round


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

def save_training_samples(samples):
    np.save('samples.npy', samples)

def load_samples(filename):
    samples = np.load(filename)
    return samples
