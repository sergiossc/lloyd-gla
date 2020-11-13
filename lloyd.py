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
    percentage_of_sub_samples = parm['percentage_of_sub_samples']
    initial_alphabet_method = parm['initial_alphabet_method']

    # Saving some information on data dict to (in the end) put it in json file
    data['num_of_elements'] = num_of_elements
    data['variance_of_samples'] = variance_of_samples
    data['use_same_samples_for_all'] = use_same_samples_for_all
    data['initial_alphabet_opt'] = initial_alphabet_opt
    data['distortion_measure_opt'] = distortion_measure_opt
    data['num_of_samples'] = num_of_samples
    data['num_of_interactions'] = num_of_interactions
    data['percentage_of_sub_samples'] = percentage_of_sub_samples
    data['initial_alphabet_method'] = initial_alphabet_method

    dftcodebook = gen_dftcodebook(num_of_elements)

    #dftcodebook = np.array([cw for cw  in dftcodebook])
    #plot_codebook(dftcodebook, 'dftcodebook.png')

    data['dftcodebook'] = encode_codebook(matrix2dict(dftcodebook))
    
    ##use_same_samples_for_all = d['use_same_samples_for_all']
    samples = gen_samples(dftcodebook, num_of_samples, variance_of_samples, use_same_samples_for_all)
   
    num_samples, num_rows, num_cols = samples.shape
    initial_codebook = katsavounidis_initial_codebook(samples)
    plot_codebook(initial_codebook, 'initial_codebook_from_katsavounidis_initial_codebook.png')
#    initial_codebook = np.zeros((num_of_elements, num_rows, num_cols), dtype=complex)
#     
#    if initial_alphabet_opt == 'user_defined':
#        if initial_alphabet_opt == 'xiaoxiao':
#            initial_codebook, samples_hadamard = xiaoxiao_initial_codebook(samples)
#            samples = samples_hadamard
#        elif initial_alphabet_opt == 'katsavounidis':
#            initial_codebook = katsavounidis_initial_codebook(samples)
#    
#    #    #print ('max_distance: ', max_distance)
#    #print ('initial_codebook: \n', initial_codebook)
#    #    max_sample = max_distance_sample
#
#    #plot_codebook(initial_codebook, 'initial_codebook_from_xiaoxiao_paper.png')
#
#    #samples_avg = complex_average(samples)
#
#    ##samples = [richscatteringchnmtx(num_of_elements, 1, variance_of_samples) for i in range(num_of_samples)]
#    ##samples = np.array(samples)
#    
#    #samples_normalized = np.array([sample/norm(sample) for sample  in samples])
#    #samples_sorted, attr_sorted = sorted_samples(samples, 'stddev')
#
#    ##samples_sorted_avg, attr_sorted_avg = sorted_samples(samples, 'avg_xiaoxiao')
#    ##samples_sorted_var, attr_sorted_var = sorted_samples(samples, 'var_xiaoxiao')
#    
#    ##plot_filename_avg = 'plot_samples_avg_nt16_ordered_' + str(instance_id) + '.png'
#    ##plot_filename_var = 'plot_samples_var_nt16_ordered_' + str(instance_id) + '.png'
#
#    #plot_samples(attr_sorted_avg, plot_filename_avg, r'$abs(m_x)$', 'abs(m_x)')
#    ##plot_samples(attr_sorted_avg, plot_filename_avg, r'Samples in ascending order by $abs(m_x)$: $N_r = 1$, $N_t = 16$, $k = $' + str(num_of_samples), r'$abs(m_x)$')
#    ##plot_samples(attr_sorted_var, plot_filename_var, r'Samples in ascending order by $var_x$: $N_r = 1$, $N_t = 16$, $k = $' + str(num_of_samples), r'$var_x$')
#
#    # Here, the number of lloyd levels or reconstruction alphabet is equal to number of elements
#    num_of_levels = num_of_elements
#    data['num_of_levels'] = num_of_levels
#
#    # Choose a seed to keep a track of this trial. This seed is saved on json data file.
#    trial_seed = np.random.randint(5, 500000)
#    np.random.seed(trial_seed)
#    data['random_seed'] = trial_seed
# 
#    # Setup is ready! Now I can run lloyd algotihm according to the initial alphabet option chosen
#    lloydcodebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, num_of_interactions, distortion_measure_opt, variance_of_samples, initial_codebook, percentage_of_sub_samples)
#
#    ##plot_performance(mean_distortion_by_round, 'MSE as distortion', 'distortion.png')
#    if initial_alphabet_opt == 'user_defined':
#        if initial_alphabet_opt == 'xiaoxiao':
#            # We have to get inverse transform from hadamard code
#            lloydcodebook = hadamard_transform(lloydcodebook, True)
#        elif initial_alphabet_opt == 'katsavounidis':
#            pass
# 
#    data['lloydcodebook'] = encode_codebook(matrix2dict(lloydcodebook))
#    data['sets'] = encode_sets(sets)
#    data['mean_distortion_by_round'] = encode_mean_distortion(mean_distortion_by_round)
#
#    with open(json_filename, "w") as write_file:
#        json.dump(data, write_file, indent=4)

    return 0

    
if __name__ == '__main__':
    profile_pathfile = 'profile.json' 

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
    percentage_of_sub_samples = d['percentage_of_sub_samples']
    initial_alphabet_method = d['initial_alphabet_method']

    parms = []
    for n_elements in num_of_elements:
        for variance in variance_of_samples_values:
            for initial_alphabet_opt in initial_alphabet_opts:
                for distortion_measure_opt in distortion_measure_opts:
                    for initial_alphabet_method_opt in initial_alphabet_method:
                        for n in range(num_of_trials):
                            p = {'num_of_elements': n_elements, 'variance_of_samples': variance, 'initial_alphabet_opt':initial_alphabet_opt, 'distortion_measure_opt':distortion_measure_opt, 'num_of_samples':num_of_samples, 'num_of_interactions':num_of_interactions, 'results_dir': results_dir, 'use_same_samples_for_all': use_same_samples_for_all, 'instance_id': str(uuid.uuid4()), 'percentage_of_sub_samples': percentage_of_sub_samples, 'initial_alphabet_method': initial_alphabet_method_opt}
                            parms.append(p)
    
    print ('# of cpus: ', os.cpu_count())
    print ('# of parms: ', len(parms))
    
    with concurrent.futures.ProcessPoolExecutor() as e:
        for p, r in zip(parms, e.map(run_lloyd_gla, parms)):
            print ('parm ' + str(p['instance_id']) + ' returned  ' + str(r))
