#!/usr/bin/env python3
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

def run_lloyd_gla(parm):
    instance_id = parm['instance_id']
    results_dir = parm['results_dir']
    json_filename = str(results_dir) + '/' + str(instance_id) + '.json'

    data = {}
    data['instance_id'] = str(instance_id)

    # Getting information from params
    num_of_elements = parm['num_of_elements']
    variance_of_samples = parm['variance_of_samples']
    use_same_samples_for_all = parm['use_same_samples_for_all']
    initial_alphabet_opt = parm['initial_alphabet_opt']
    distortion_measure_opt = parm['distortion_measure_opt']
    num_of_samples = parm['num_of_samples']
    num_of_interactions = parm['num_of_interactions']

    # Saving some information on data dict to put it in json file
    data['num_of_elements'] = num_of_elements
    data['variance_of_samples'] = variance_of_samples
    data['use_same_samples_for_all'] = use_same_samples_for_all
    data['initial_alphabet_opt'] = initial_alphabet_opt
    data['distortion_measure_opt'] = distortion_measure_opt
    data['num_of_samples'] = num_of_samples
    data['num_of_interactions'] = num_of_interactions


    dftcodebook = gen_dftcodebook(num_of_elements)
    data['dftcodebook'] = encode_codebook(matrix2dict(dftcodebook))

    use_same_samples_for_all = d['use_same_samples_for_all']
    samples = gen_samples(dftcodebook, num_of_samples, variance_of_samples, use_same_samples_for_all)

    # Getting samples mixed
    samples_index = np.random.choice(len(samples), len(samples), replace=False)
    samples_mixed = np.array([samples[i] for i in samples_index]) 
    
    # Getting samples as unitary vectors dividing by its own norm
    samples_normalized = np.array([sample/norm(sample) for sample  in samples_mixed])

    # Here, the number of lloyd levels or reconstruction alphabet is equal to number of elements
    num_of_levels = num_of_elements
    data['num_of_levels'] = num_of_levels

    # Choose a seed to keep a track from trial. This seed is saved on json data file.
    trial_seed = np.random.randint(5, 500000)
    np.random.seed(trial_seed)
    data['random_seed'] = trial_seed
 
    # Setup is ready! Now I can run lloyd algotihm according to the initial alphabet option chosen

    if initial_alphabet_opt == 'sa':
        initial_codebook = np.array([samples_normalized[i] for i in np.random.choice(len(samples), num_of_levels, replace=False)])
        initial_temperature = 10
        max_sa_num_of_iteractions = 10
        lloydcodebook, sets, mean_distortion_by_round = sa(initial_codebook, variance_of_samples, initial_temperature, max_sa_num_of_iteractions, num_of_interactions, distortion_measure_opt, num_of_levels, samples_normalized)
    else:
        lloydcodebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples_normalized, num_of_levels, num_of_interactions, distortion_measure_opt, variance_of_samples, initial_codebook)

    data['lloydcodebook'] = encode_codebook(matrix2dict(lloydcodebook))
    data['sets'] = encode_sets(sets)
    data['mean_distortion_by_round'] = encode_mean_distortion(mean_distortion_by_round)

    with open(json_filename, "w") as write_file:
        json.dump(data, write_file, indent=4)

    return 0

def sa(initial_codebook, variance_of_samples, initial_temperature, max_iteractions, lloyd_num_of_interactions, distortion_measure_opt, num_of_levels, samples):
    
    best_lloydcodebook, best_sets, best_mean_distortion_by_round = lloyd_gla("sa", samples, num_of_levels, lloyd_num_of_interactions, distortion_measure_opt, variance_of_samples, initial_codebook)
    best_mean_distortion_list = list(best_mean_distortion_by_round[1])
    best_distortion = best_mean_distortion_list[-1]
    current_temperature = initial_temperature
    current_iteraction = 0
    while current_temperature > 0.01:
        print (current_temperature)
        while current_iteraction < max_iteractions:
            
            candidate_codebook = gen_samples(initial_codebook, num_of_levels, variance, False)
            candidate_lloydcodebook, candidate_sets, candidate_mean_distortion_by_round = lloyd_gla("sa", samples, num_of_levels, lloyd_num_of_interactions, distortion_measure_opt, variance_of_samples, candidate_codebook)
            candidate_distortion_by_lloyd_interactions = list(candidate_mean_distortion_by_round[1])
            candidate_distortion = candidate_distortion_by_lloyd_interactions[-1]


            initial_lloydcodebook, initial_sets, initial_mean_distortion_by_round = lloyd_gla("sa", samples, num_of_levels, lloyd_num_of_interactions, distortion_measure_opt, variance_of_samples, initial_codebook)
            initial_distortion_by_lloyd_interactions = list(initial_mean_distortion_by_round[1])
            initial_distortion = initial_distortion_by_lloyd_interactions[-1]

            delta_distortion = candidate_distortion - initial_distortion
            if delta_distortion < 0:
                initial_codebook = candidate_codebook
                if (candidate_distortion < best_distortion):
                    best_distortion = candidate_distortion
                    best_lloydcodebook = candidate_lloydcodebook
                    best_sets = candidate_sets
                    best_mean_distortion_by_round = candidate_mean_distortion_by_round
                    #print ('candidate: ', candidate_distortion)
                    #print ('initial: ', initial_distortion)
            else:
                x = np.random.rand()
                if (x < np.exp(-delta_distortion/current_temperature)):
                    initial_codebook = candidate_codebook

            current_iteraction += 1
        current_temperature = current_temperature * 0.1
        current_iteraction = 0
    print (best_distortion)
    return best_lloydcodebook, best_sets, best_mean_distortion_by_round
    
if __name__ == '__main__':
    profile_pathfile = 'profile_sa.json' 

    with open(profile_pathfile) as profile:
        data = profile.read()
        d = json.loads(data)

    # Read information from 'profile.json' file
    num_of_elements = d['number_of_elements']
    variance_of_samples_values = d['variance_of_samples_values']
    initial_alphabet_opts = d['initial_alphabet_opts']
    distortion_measure_opts = d['distortion_measure_opts']
    num_of_trials = d['num_of_trials']
    num_of_samples = d['num_of_samples']
    num_of_interactions = d['num_of_interactions']
    results_dir = d['results_directory']
    use_same_samples_for_all = d['use_same_samples_for_all']

    parms = []
    for n_elements in num_of_elements:
        for variance in variance_of_samples_values:
            for initial_alphabet_opt in initial_alphabet_opts:
                for distortion_measure_opt in distortion_measure_opts:
                    for n in range(num_of_trials):
                        p = {'num_of_elements': n_elements, 'variance_of_samples': variance, 'initial_alphabet_opt':initial_alphabet_opt, 'distortion_measure_opt':distortion_measure_opt, 'num_of_samples':num_of_samples, 'num_of_interactions':num_of_interactions, 'results_dir': results_dir, 'use_same_samples_for_all': use_same_samples_for_all, 'instance_id': str(uuid.uuid4())}
                        parms.append(p)
    
    print ('# of cpus: ', os.cpu_count())
    print ('# of parms: ', len(parms))
    
    with concurrent.futures.ProcessPoolExecutor() as e:
        for p, r in zip(parms, e.map(run_lloyd_gla, parms)):
            print ('parm ' + str(p['instance_id']) + ' returned  ' + str(r))
