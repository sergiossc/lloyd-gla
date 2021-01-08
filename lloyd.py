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
import argparse



def run_retrial(trial_pathfile):

    print('Re-trial begin_________________________________')
    print ('json pathfile: ', trial_pathfile)

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

def running_on_cluster(profile_pathfile):

    submit_pathfile = create_submit_file(profile_pathfile)

    print (submit_pathfile)


    try:

        with open(submit_pathfile) as auto_submitfile:
            command = 'condor_submit ' + str(submit_pathfile)
            os.system(command)

    except:
        print('Could not read submit file')
 



def create_submit_file(profile_pathfile):

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
  
        
    args = ''
    args = '\nexecutable = ' + str('run_lloyd.py') 
    args += '\nlog = ' + str('run_lloyd.$(Cluster).$(Process).out')
    args += '\nerror = ' + str('run_lloyd.$(Cluster).$(Process).err')
    #args += '\noutput = ' +
    args += '\nshould_transfer_files = ' + str('Yes')
    args += '\ntransfer_input_files = ' + str('utils.py')
    args += '\nwhen_to_transfer_output = ' + str('ON_EXIT')

    for p in parms:
        print ('***************')

        num_of_elements = p['num_of_elements']
        variance_of_samples = p['variance_of_samples']
        initial_alphabet_opt = p['initial_alphabet_opt']
        distortion_measure_opt = p['distortion_measure_opt']
        num_of_samples = p['num_of_samples']
        max_num_of_interactions = p['max_num_of_interactions']
        results_dir = p['results_dir']
        instance_id = p['instance_id']
        percentage_of_sub_samples = p['percentage_of_sub_samples']
        samples_random_seed = p['samples_random_seed']
        trial_random_seed = p['trial_random_seed']
    
        #args = 'run_lloyd_gla(' + str(num_of_elements) + ', ' +  str(variance_of_samples) + ', \'' + str(initial_alphabet_opt) + '\', \'' + str(distortion_measure_opt) + '\', ' + str(num_of_samples) + ', ' + str(max_num_of_interactions) + ', \'' + str(results_dir) + '\', \'' + str(instance_id) + '\', ' + str(percentage_of_sub_samples) + ', ' + str(samples_random_seed) + ', ' + str(trial_random_seed) + ')'
        args += '\narguments = ' + str(num_of_elements) + ' ' +  str(variance_of_samples) + ' ' + str(initial_alphabet_opt) + ' ' + str(distortion_measure_opt) + ' ' + str(num_of_samples) + ' ' + str(max_num_of_interactions) + ' ' + str(results_dir) + ' ' + str(instance_id) + ' ' + str(percentage_of_sub_samples) + ' ' + str(samples_random_seed) + ' ' + str(trial_random_seed) + ' \nqueue'


    submit_pathfile = 'auto-submit-v.sh'

    

    try:

        f = open(submit_pathfile, 'w+')
        f.write(args)
        f.close()
        print (args)
        return submit_pathfile

    except:
        print('Could not create submit file')
    

def running_locally(profile_pathfile):
    print (profile_pathfile)
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



if __name__ == '__main__':
    
    my_parser = argparse.ArgumentParser()
    my_group = my_parser.add_mutually_exclusive_group(required=True)

    my_group.add_argument('-r', '--retrial', action='store_true', help='retrial an specific trial from a json result file')
    my_group.add_argument('-c', '--running-cluster', action='store_true', help='allows to run all trials on cluster')
    my_group.add_argument('-l', '--running-locally', action='store_true', help='allows to run all trials locally on PC using python concurrents')

    my_parser.add_argument('-j', '--json-pathfile-result', metavar='path', type=str, help='the path to json file with some results of a trial')
    my_parser.add_argument('-p', '--profile-pathfile', metavar='path', type=str, help='the path to profile json file with general parameters')

    args = my_parser.parse_args()

    if args.retrial == True and args.running_cluster == False and args.running_locally == False:
        print ('Run a retrial!')
        #json_pathfile_result = args.json_pathfile_result
        print (args.json_pathfile_result)
        
        if args.json_pathfile_result is None:
            print ('Use \'-j path\' option')
            sys.exit()

        if not os.path.isfile(args.json_pathfile_result):
            print('The json pathfile specified does not exist')
            sys.exit()
        else:
            run_retrial(args.json_pathfile_result)


    elif args.retrial == False and args.running_cluster == True and args.running_locally == False:
        print ('Running trials on cluster!')

        if args.profile_pathfile is None:
            print ('Use \'-p path\' option to indicate the profile pathfile')
            sys.exit()

        if not os.path.isfile(args.profile_pathfile):
            print('The profile pathfile specified does not exist')
            sys.exit()
        else:
            running_on_cluster(args.profile_pathfile)


    elif args.retrial == False and args.running_cluster == False and args.running_locally == True:
        print ('Running trials locally')

        if args.profile_pathfile is None:
            print ('Use \'-p path\' option to indicate the profile pathfile')
            sys.exit()

        if not os.path.isfile(args.profile_pathfile):
            print('The profile pathfile specified does not exist')
            sys.exit()
        else:
            running_locally(args.profile_pathfile)

    else:
        #pass
        print ('Use \'-h\' option to see available options')
        sys.exit()

    #print(vars(args))
    
