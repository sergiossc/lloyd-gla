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
    use_same_samples_for_all = parm['use_same_samples_for_all']
    initial_alphabet_opt = parm['initial_alphabet_opt']
    distortion_measure_opt = parm['distortion_measure_opt']
    num_of_samples = parm['num_of_samples']
    max_num_of_interactions = parm['max_num_of_interactions']
    percentage_of_sub_samples = parm['percentage_of_sub_samples']
    #initial_alphabet_method = parm['initial_alphabet_method']
    seed = parm['samples_random_seed']

    # Saving some information on data dict to (in the end) put it in json file
    data['num_of_elements'] = num_of_elements
    data['variance_of_samples'] = variance_of_samples
    data['use_same_samples_for_all'] = use_same_samples_for_all
    data['samples_random_seed'] = float(seed)
    data['initial_alphabet_opt'] = initial_alphabet_opt
    data['distortion_measure_opt'] = distortion_measure_opt
    data['num_of_samples'] = num_of_samples
    data['max_num_of_interactions'] = max_num_of_interactions
    data['percentage_of_sub_samples'] = percentage_of_sub_samples
    #data['initial_alphabet_method'] = initial_alphabet_method

    dftcodebook = gen_dftcodebook(num_of_elements)

    #dftcodebook = np.array([cw for cw  in dftcodebook])
    #plot_codebook(dftcodebook, 'dftcodebook.png')

    data['dftcodebook'] = encode_codebook(matrix2dict(dftcodebook))
    
    # Here, the number of lloyd levels or reconstruction alphabet is equal to number of elements
    num_of_levels = num_of_elements

    # Creating samples
    samples = gen_samples(dftcodebook, num_of_samples, variance_of_samples, seed)
   
    num_samples, num_rows, num_cols = samples.shape
    #initial_codebook = katsavounidis_initial_codebook(samples)
    #print (initial_codebook.shape)
    #plot_codebook(initial_codebook, 'my_initial_codebook_from_katsavounidis_initial_codebook.png')
    #initial_codebook = np.zeros((num_of_levels, num_rows, num_cols), dtype=complex)
     
    if initial_alphabet_opt == 'xiaoxiao':

        initial_codebook, samples_hadamard = xiaoxiao_initial_codebook(samples)
        samples = samples_hadamard
        data['initial_codebook'] = encode_codebook(matrix2dict(initial_codebook))
        lloydcodebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, None, initial_codebook, None)
        #lloydcodebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, variance_of_samples, initial_codebook, percentage_of_sub_samples)
        lloydcodebook = hadamard_transform(lloydcodebook, True)

    elif initial_alphabet_opt == 'katsavounidis':

        initial_codebook = katsavounidis_initial_codebook(samples)
        data['initial_codebook'] = encode_codebook(matrix2dict(initial_codebook))
        lloydcodebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, None, initial_codebook, None)
        #lloydcodebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, variance_of_samples, initial_codebook, percentage_of_sub_samples)

    elif initial_alphabet_opt == 'sa':
        initial_codebook = np.array([samples[i] for i in np.random.choice(len(samples), num_of_levels, replace=False)])
        initial_temperature = 10
        sa_max_num_of_iteractions = 20
        lloydcodebook, sets, mean_distortion_by_round = sa(initial_codebook, variance_of_samples, initial_temperature, sa_max_num_of_iteractions, max_num_of_interactions, distortion_measure_opt, num_of_levels, samples)

    elif initial_alphabet_opt == 'unitary_until_num_of_elements':
        lloydcodebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, variance_of_samples, None, percentage_of_sub_samples)

    elif initial_alphabet_opt == 'random_from_samples':
        lloydcodebook, sets, mean_distortion_by_round = lloyd_gla(initial_alphabet_opt, samples, num_of_levels, max_num_of_interactions, distortion_measure_opt, variance_of_samples, None, percentage_of_sub_samples)
 

    #    #print ('max_distance: ', max_distance)
    #print ('initial_codebook: \n', initial_codebook)
    #    max_sample = max_distance_sample

    #plot_codebook(initial_codebook, 'initial_codebook_from_' + str(initial_alphabet_method) + '_' + str(instance_id) + '_paper.png')

    #samples_avg = complex_average(samples)

    ##samples = [richscatteringchnmtx(num_of_elements, 1, variance_of_samples) for i in range(num_of_samples)]
    ##samples = np.array(samples)
    
    #samples_normalized = np.array([sample/norm(sample) for sample  in samples])
    #samples_sorted, attr_sorted = sorted_samples(samples, 'stddev')

    ##samples_sorted_avg, attr_sorted_avg = sorted_samples(samples, 'avg_xiaoxiao')
    ##samples_sorted_var, attr_sorted_var = sorted_samples(samples, 'var_xiaoxiao')
    
    ##plot_filename_avg = 'plot_samples_avg_nt16_ordered_' + str(instance_id) + '.png'
    ##plot_filename_var = 'plot_samples_var_nt16_ordered_' + str(instance_id) + '.png'

    #plot_samples(attr_sorted_avg, plot_filename_avg, r'$abs(m_x)$', 'abs(m_x)')
    ##plot_samples(attr_sorted_avg, plot_filename_avg, r'Samples in ascending order by $abs(m_x)$: $N_r = 1$, $N_t = 16$, $k = $' + str(num_of_samples), r'$abs(m_x)$')
    ##plot_samples(attr_sorted_var, plot_filename_var, r'Samples in ascending order by $var_x$: $N_r = 1$, $N_t = 16$, $k = $' + str(num_of_samples), r'$var_x$')




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
            print('trial begin_________________________________')

            with open(trial_pathfile) as trial_results:
                data = trial_results.read()
                d = json.loads(data)
        
            # Read information from 'profile.json' file
            instance_id = 'retrial_of_' + d['instance_id']
            n_elements = d['num_of_elements']
            variance = d['variance_of_samples']
            initial_alphabet_opt = d['initial_alphabet_opt']
            distortion_measure_opt = d['distortion_measure_opt']
            num_of_samples = d['num_of_samples']
            max_num_of_interactions = d['max_num_of_interactions']
            results_dir = d['results_dir']
            use_same_samples_for_all = d['use_same_samples_for_all']
            percentage_of_sub_samples = d['percentage_of_sub_samples']


            samples_random_seed = d['samples_random_seed']
            #initial_alphabet_method = d['initial_alphabet_method']

            #dftcodebook = dict2matrix(decode_codebook(d['dftcodebook']))
            #nrows, ncols = dftcodebook.shape

            #initial_codebook = np.zeros((nrows, 1, ncols),dtype=complex)
            #if initial_alphabet_opt == 'user_defined':
            #    initial_codebook = dict2matrix(decode_codebook(d['initial_codebook'])) 
            #    initial_codebook = np.array(initial_codebook).reshape(nrows, 1, ncols)
            #print (initial_codebook.shape)
                                
            p = {'num_of_elements': n_elements, 'variance_of_samples': variance, 'initial_alphabet_opt':initial_alphabet_opt, 'distortion_measure_opt':distortion_measure_opt, 'num_of_samples':num_of_samples, 'max_num_of_interactions':max_num_of_interactions, 'results_dir': results_dir, 'use_same_samples_for_all': use_same_samples_for_all, 'instance_id': instance_id, 'percentage_of_sub_samples': percentage_of_sub_samples, 'samples_random_seed': int(samples_random_seed)}

            run_lloyd_gla(p)

            print (p)
            print('trial end_________________________________')


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
        #initial_alphabet_method = d['initial_alphabet_method']
    
        parms = []
        for n_elements in num_of_elements:
            for variance in variance_of_samples_values:
                for initial_alphabet_opt in initial_alphabet_opts:
                    for distortion_measure_opt in distortion_measure_opts:
                        #for initial_alphabet_method_opt in initial_alphabet_method:
                        for n in range(num_of_trials):
                            p = {'num_of_elements': n_elements, 'variance_of_samples': variance, 'initial_alphabet_opt':initial_alphabet_opt, 'distortion_measure_opt':distortion_measure_opt, 'num_of_samples':num_of_samples, 'max_num_of_interactions':max_num_of_interactions, 'results_dir': results_dir, 'use_same_samples_for_all': use_same_samples_for_all, 'instance_id': str(uuid.uuid4()), 'percentage_of_sub_samples': percentage_of_sub_samples}
                            parms.append(p)
        
        if use_same_samples_for_all:
            random_seeds = np.ones(len(parms)) * 789
            random_seeds = np.array([int(v) for v in random_seeds])
        else: 
            random_seeds = np.random.choice(100000, len(parms), replace=False)
        
        for n in range(len(parms)): 
            p = parms[n]
            seed = random_seeds[n]
            p['samples_random_seed'] = seed
        
        print ('# of cpus: ', os.cpu_count())
        print ('# of parms: ', len(parms))
        
        with concurrent.futures.ProcessPoolExecutor() as e:
            for p, r in zip(parms, e.map(run_lloyd_gla, parms)):
                print ('parm ' + str(p['instance_id']) + ' returned  ' + str(r))
