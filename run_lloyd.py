#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
@author: sergiossc@gmail.com
"""
import concurrent.futures
import numpy as np
import uuid
import json
import os
from utils import *
import sys


#def run_lloyd_gla(num_of_elements, variance_of_samples, initial_alphabet_opt, distortion_measure_opt, num_of_samples, max_num_of_interactions, results_dir, instance_id, percentage_of_sub_samples, samples_random_seed, trial_random_seed):
#
#    #instance_id = instance_id
#    #results_dir = results_dir
#    #json_filename = str(results_dir) + '/' + str(instance_id) + '.json'
#    json_filename = str(instance_id) + '.json'
#
#    data = {}
#    data['instance_id'] = str(instance_id)
#    data['results_dir'] = results_dir
#
#    # Saving some information on data dict to (in the end) put it in json file
#    data['num_of_elements'] = num_of_elements
#    data['variance_of_samples'] = variance_of_samples
#    data['samples_random_seed'] = float(samples_random_seed)
#    data['trial_random_seed'] = float(trial_random_seed)
#    data['initial_alphabet_opt'] = initial_alphabet_opt
#    data['distortion_measure_opt'] = distortion_measure_opt
#    data['num_of_samples'] = num_of_samples
#    data['max_num_of_interactions'] = max_num_of_interactions
#    data['percentage_of_sub_samples'] = percentage_of_sub_samples
#
#    dftcodebook = gen_dftcodebook(num_of_elements)
#    num_of_levels = num_of_elements
#
#    data['dftcodebook'] = encode_codebook(matrix2dict(dftcodebook))
#    
#    # Here, the number of lloyd levels or reconstruction alphabet is equal to number of elements
#    #num_of_levels = num_of_elements
#
#    # Creating samples
#    samples = gen_samples(dftcodebook, num_of_samples, variance_of_samples, samples_random_seed)
#
#    # Controlling randomness from trial by seed to make possible reproduce it later
#    np.random.seed(trial_random_seed)
#    lloydcodebook = np.zeros((num_of_samples, num_of_samples), dtype=complex)
#   
#    # Starting lloyd with an specific initial alphabet opt
#    if initial_alphabet_opt == 'xiaoxiao':
#
#        initial_codebook, samples_hadamard = xiaoxiao_initial_codebook(samples)
#        #samples = samples_hadamard
#        data['initial_codebook'] = encode_codebook(matrix2dict(initial_codebook))
#        lloydcodebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, None, initial_codebook, None)
#        #lloydcodebook = hadamard_transform(lloydcodebook, True)
#
#    elif initial_alphabet_opt == 'katsavounidis':
#
#        initial_codebook = katsavounidis_initial_codebook(samples)
#        data['initial_codebook'] = encode_codebook(matrix2dict(initial_codebook))
#        lloydcodebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, None, initial_codebook, None)
#
#    elif initial_alphabet_opt == 'sa':
#        initial_codebook = np.array([samples[i] for i in np.random.choice(len(samples), num_of_levels, replace=False)])
#        data['initial_codebook'] = encode_codebook(matrix2dict(initial_codebook))
#        initial_temperature = 10
#        sa_max_num_of_iteractions = 20
#        lloydcodebook, sets, mean_distortion_by_round = sa(initial_codebook, variance_of_samples, initial_temperature, sa_max_num_of_iteractions, max_num_of_interactions, distortion_measure_opt, num_of_levels, samples)
#
#    elif initial_alphabet_opt == 'unitary_until_num_of_elements':
#        initial_codebook = complex_average(samples)
#        data['initial_codebook'] = encode_codebook(matrix2dict(initial_codebook))
#        lloydcodebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, variance_of_samples, initial_codebook, percentage_of_sub_samples)
#
#    elif initial_alphabet_opt == 'random_from_samples':
#        initial_codebook = np.array([samples[i] for i in np.random.choice(len(samples), num_of_levels, replace=False)])
#        data['initial_codebook'] = encode_codebook(matrix2dict(initial_codebook))
#        lloydcodebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, variance_of_samples, initial_codebook, percentage_of_sub_samples)
# 
#
#    ##plot_performance(mean_distortion_by_round, 'MSE as distortion', 'distortion.png')
#
#    # Saving results in JSON file 
#    data['lloydcodebook'] = encode_codebook(matrix2dict(lloydcodebook))
#    data['sets'] = encode_sets(sets)
#    data['mean_distortion_by_round'] = encode_mean_distortion(mean_distortion_by_round)
#
#    with open(json_filename, "w") as write_file:
#        json.dump(data, write_file, indent=4)
#
#    return 0

    
if __name__ == '__main__':

    
     num_of_elements = int(sys.argv[1])
     variance_of_samples = float(sys.argv[2])
     initial_alphabet_opt = sys.argv[3]
     distortion_measure_opt = sys.argv[4]
     num_of_samples = int(sys.argv[5])
     max_num_of_interactions = int(sys.argv[6])
     results_dir = sys.argv[7]
     instance_id = sys.argv[8]
     percentage_of_sub_samples = float(sys.argv[9])
     samples_random_seed = int(sys.argv[10])
     trial_random_seed = int(sys.argv[11])

     p = {'num_of_elements': num_of_elements, 'variance_of_samples': variance_of_samples, 'initial_alphabet_opt':initial_alphabet_opt, 'distortion_measure_opt':distortion_measure_opt, 'num_of_samples':num_of_samples, 'max_num_of_interactions':max_num_of_interactions, 'results_dir': results_dir, 'instance_id': instance_id, 'percentage_of_sub_samples': percentage_of_sub_samples, 'samples_random_seed': samples_random_seed, 'trial_random_seed': trial_random_seed}

     result = run_lloyd_gla(p)

     #result = run_lloyd_gla(num_of_elements,
     #        variance_of_samples, 
     #        initial_alphabet_opt, 
     #        distortion_measure_opt, 
     #        num_of_samples, 
     #        max_num_of_interactions, 
     #        results_dir, 
     #        instance_id, 
     #        percentage_of_sub_samples, 
     #        samples_random_seed, 
     #        trial_random_seed)

