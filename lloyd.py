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
import sys

def run_lloyd_gla(parm):
    instance_id = parm['instance_id']
    results_dir = parm['results_dir']
    json_filename = str(results_dir) + '/' + str(instance_id) + '.json'

    data = {}
    data['instance_id'] = str(instance_id)
    data['results_dir'] = results_dir

    # Getting information from params
    num_of_elements = parm['num_of_elements']
    variance_of_samples = parm['variance_of_samples']
    initial_alphabet_opt = parm['initial_alphabet_opt']
    distortion_measure_opt = parm['distortion_measure_opt']
    num_of_samples = parm['num_of_samples']
    max_num_of_interactions = parm['max_num_of_interactions']
    percentage_of_sub_samples = parm['percentage_of_sub_samples']
    samples_random_seed = parm['samples_random_seed']
    trial_random_seed = parm['trial_random_seed']

    # Saving some information on data dict to (in the end) put it in json file
    data['num_of_elements'] = num_of_elements
    data['variance_of_samples'] = variance_of_samples
    data['samples_random_seed'] = float(samples_random_seed)
    data['trial_random_seed'] = float(trial_random_seed)
    data['initial_alphabet_opt'] = initial_alphabet_opt
    data['distortion_measure_opt'] = distortion_measure_opt
    data['num_of_samples'] = num_of_samples
    data['max_num_of_interactions'] = max_num_of_interactions
    data['percentage_of_sub_samples'] = percentage_of_sub_samples

    dftcodebook = gen_dftcodebook(num_of_elements)

    data['dftcodebook'] = encode_codebook(matrix2dict(dftcodebook))
    
    # Here, the number of lloyd levels or reconstruction alphabet is equal to number of elements
    num_of_levels = num_of_elements

    # Creating samples
    samples = gen_samples(dftcodebook, num_of_samples, variance_of_samples, samples_random_seed)

    # Controlling randomness from trial by seed to make possible reproduce it later
    np.random.seed(trial_random_seed)
   
    # Starting lloyd with an specific initial alphabet opt
    if initial_alphabet_opt == 'xiaoxiao':

        initial_codebook, samples_hadamard = xiaoxiao_initial_codebook(samples)
        #samples = samples_hadamard
        data['initial_codebook'] = encode_codebook(matrix2dict(initial_codebook))
        lloydcodebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, None, initial_codebook, None)
        #lloydcodebook = hadamard_transform(lloydcodebook, True)

    elif initial_alphabet_opt == 'katsavounidis':

        initial_codebook = katsavounidis_initial_codebook(samples)
        data['initial_codebook'] = encode_codebook(matrix2dict(initial_codebook))
        lloydcodebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, None, initial_codebook, None)

    elif initial_alphabet_opt == 'sa':
        initial_codebook = np.array([samples[i] for i in np.random.choice(len(samples), num_of_levels, replace=False)])
        data['initial_codebook'] = encode_codebook(matrix2dict(initial_codebook))
        initial_temperature = 10
        sa_max_num_of_iteractions = 20
        lloydcodebook, sets, mean_distortion_by_round = sa(initial_codebook, variance_of_samples, initial_temperature, sa_max_num_of_iteractions, max_num_of_interactions, distortion_measure_opt, num_of_levels, samples)

    elif initial_alphabet_opt == 'unitary_until_num_of_elements':
        initial_codebook = complex_average(samples)
        data['initial_codebook'] = encode_codebook(matrix2dict(initial_codebook))
        lloydcodebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, variance_of_samples, initial_codebook, percentage_of_sub_samples)

    elif initial_alphabet_opt == 'random_from_samples':
        initial_codebook = np.array([samples[i] for i in np.random.choice(len(samples), num_of_levels, replace=False)])
        data['initial_codebook'] = encode_codebook(matrix2dict(initial_codebook))
        lloydcodebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, variance_of_samples, initial_codebook, percentage_of_sub_samples)
 

    ##plot_performance(mean_distortion_by_round, 'MSE as distortion', 'distortion.png')

    # Saving results in JSON file 
    data['lloydcodebook'] = encode_codebook(matrix2dict(lloydcodebook))
    data['sets'] = encode_sets(sets)
    data['mean_distortion_by_round'] = encode_mean_distortion(mean_distortion_by_round)

    with open(json_filename, "w") as write_file:
        json.dump(data, write_file, indent=4)

    return 0

    
if __name__ == '__main__':

    command_line_parms_len = len(sys.argv)

    if command_line_parms_len > 1:
        trial_pathfile = sys.argv[1]
        if not os.path.isfile(trial_pathfile):
            print('Wrong trial pathfile')
        else:
            print('Re-trial begin_________________________________')

            with open(trial_pathfile) as trial_results:
                data = trial_results.read()
                d = json.loads(data)
        
            # Read information from 'profile.json' file
            instance_id = 'retrial_of_' + d['instance_id']
            print (instance_id)
            n_elements = d['num_of_elements']
            variance = d['variance_of_samples']
            initial_alphabet_opt = d['initial_alphabet_opt']
            distortion_measure_opt = d['distortion_measure_opt']
            num_of_samples = d['num_of_samples']
            max_num_of_interactions = d['max_num_of_interactions']
            results_dir = d['results_dir']
            #use_same_samples_for_all = d['use_same_samples_for_all']
            percentage_of_sub_samples = d['percentage_of_sub_samples']


            samples_random_seed = d['samples_random_seed']
            trial_random_seed = d['trial_random_seed']


            lloydcodebook_dict = decode_codebook(d['lloydcodebook'])
            initialcodebook_dict = decode_codebook(d['initial_codebook'])

            lloydcodebook_matrix = dict2matrix(lloydcodebook_dict)
            initialcodebook_matrix = dict2matrix(initialcodebook_dict)

            lloyd_nrows, lloyd_ncols = lloydcodebook_matrix.shape
            lloydcodebook_matrix = np.array(lloydcodebook_matrix).reshape(lloyd_nrows, 1, lloyd_ncols)
            plot_lloydcodebook_filename =  'lloyd_codebook_from_'+ instance_id  +'.png'
            print('-- plot lloyd final codebook saved in: ', plot_lloydcodebook_filename)
            plot_codebook(lloydcodebook_matrix, plot_lloydcodebook_filename)
            print ('-- done!')

            initial_nrows, initial_ncols = initialcodebook_matrix.shape
            initialcodebook_matrix = np.array(initialcodebook_matrix).reshape(initial_nrows, 1, initial_ncols)
            plot_initialcodebook_filename = 'initial_codebook_from_' + instance_id + '.png'
            print ('-- plot initial codiedbook saved in: ', plot_initialcodebook_filename)
            plot_codebook(initialcodebook_matrix, plot_initialcodebook_filename)
            print ('-- done!')


            p = {'num_of_elements': n_elements, 'variance_of_samples': variance, 'initial_alphabet_opt':initial_alphabet_opt, 'distortion_measure_opt':distortion_measure_opt, 'num_of_samples':num_of_samples, 'max_num_of_interactions':max_num_of_interactions, 'results_dir': results_dir, 'instance_id': instance_id, 'percentage_of_sub_samples': percentage_of_sub_samples, 'samples_random_seed': int(samples_random_seed), 'trial_random_seed': int(trial_random_seed)}

            print ('running it again... ')
            run_lloyd_gla(p)
            print ('done!')

            print (p)

            print('Re-trial end_________________________________')


    else:  
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
        max_num_of_interactions = d['max_num_of_interactions']
        results_dir = d['results_directory'] 
        use_same_samples_for_all = d['use_same_samples_for_all']
        percentage_of_sub_samples = d['percentage_of_sub_samples']
    
        parms = []
        for n_elements in num_of_elements:
            for variance in variance_of_samples_values:
                for initial_alphabet_opt in initial_alphabet_opts:
                    for distortion_measure_opt in distortion_measure_opts:

                        if use_same_samples_for_all:
                            np.random.seed(789)
                            random_seeds = np.random.choice(100000, num_of_trials, replace=False)
                            np.random.seed(None)
                        else:
                            random_seeds = np.random.choice(100000, num_of_trials, replace=False)
                            
                        for n in range(num_of_trials):
                            p = {'num_of_elements': n_elements, 'variance_of_samples': variance, 'initial_alphabet_opt':initial_alphabet_opt, 'distortion_measure_opt':distortion_measure_opt, 'num_of_samples':num_of_samples, 'max_num_of_interactions':max_num_of_interactions, 'results_dir': results_dir, 'instance_id': str(uuid.uuid4()), 'percentage_of_sub_samples': percentage_of_sub_samples, 'samples_random_seed': random_seeds[n], 'trial_random_seed': np.random.choice(10000, 1)[0]}
                            parms.append(p)
        
        
        print ('# of cpus: ', os.cpu_count())
        print ('# of parms: ', len(parms))
        
        with concurrent.futures.ProcessPoolExecutor() as e:
            for p, r in zip(parms, e.map(run_lloyd_gla, parms)):
                print ('parm ' + str(p['instance_id']) + ' returned  ' + str(r))
