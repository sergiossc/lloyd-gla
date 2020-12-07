import uuid
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from utils import *

#def check_files(prefix, episodefiles):
#    pathfiles = {}
#    for ep_file in episodefiles:
#        pathfile = prefix + str('/') + str(ep_file)
#        ep_file_status = False
#        try:
#            current_file = open(pathfile)
#            ep_file_status = True
#            #print("Sucess.")
#        except IOError:
#            print("File not accessible: ", pathfile)
#        finally:
#            current_file.close()
#
#        if ep_file_status:
#            ep_file_id = uuid.uuid4()
#            pathfiles[ep_file_id] = pathfile
# 
#    return pathfiles
#
#
#def decode_mean_distortion(mean_distortion_dict):
#    mean_distortion_list = []
#    for iteration, mean_distortion in mean_distortion_dict.items():
#        mean_distortion_list.append(mean_distortion)
#    return mean_distortion_list


if __name__ == '__main__':

    profile_pathfile = 'profile.json' 

    with open(profile_pathfile) as profile:
        data = profile.read()
        d = json.loads(data)

    prefix_pathfiles = d['results_directory']
    result_files = os.listdir(prefix_pathfiles)
    pathfiles = check_files(prefix_pathfiles, result_files)
    print ('# of json files: ', len(pathfiles))
    # From here it is going to open each json file to see each parameters and data from algorithm perform. May you should to implement some decode or transate functions to deal with json data from files to python data format. There are some decode functions on utils library. 
    #trial_result = (initial_alphabet_opt, distortion_measure_opt, num_of_levels, variance_of_samples, norm)

    occurences = []
    samples_random_seeds = {}

    katsavounidis_results = {}
    xiaoxiao_results = {}
    unitary_until_num_of_elements_results = {}
    random_from_samples_results = {}
    sa_results = {}

    for pathfile_id, pathfile in pathfiles.items():
        with open(pathfile) as result:
            data = result.read()
            d = json.loads(data)

        # May you edit from right here! Tip: Read *json file in results to decide how to deal from here.
        initial_alphabet_opt = d['initial_alphabet_opt']
        variance_of_samples = d['variance_of_samples']
        distortion_measure_opt = d['distortion_measure_opt']
        initial_alphabet_opt = d['initial_alphabet_opt']
        num_of_elements = d['num_of_elements']
        num_of_levels = num_of_elements 
        num_of_samples = d['num_of_samples']
        samples_random_seed = d['samples_random_seed']
        mean_distortion_by_round = d['mean_distortion_by_round']

        #normal_vector = np.ones(num_of_levels) * (num_of_samples/num_of_levels)
        #sets = d['sets']
        #set_vector = []
        #for k, v in sets.items():
        #    set_vector.append(v)
        #set_vector = np.array(set_vector)
   
        #norm =  np.sqrt(np.sum(np.power(np.abs(set_vector - normal_vector), 2)))
        #if norm == 0 and num_of_elements == 9 and variance_of_samples == 1.0 and initial_alphabet_method == 'katsavounidis': 
        #if  norm == 0 and num_of_elements == 4 and variance_of_samples == 0.1 and initial_alphabet_method == 'katsavounidis'
        #if  variance_of_samples == 0.1 and num_of_elements == 4 and initial_alphabet_opt == 'katsavounidis':
            #trial_info = {'norm': norm}
            #occurences.append(trial_info)
        #if  num_of_elements == 4:
        samples_random_seeds[int(samples_random_seed)] = 1

        if initial_alphabet_opt == 'katsavounidis' and distortion_measure_opt == 'mse':
            last_k = ''
            for k in mean_distortion_by_round.keys():
                last_k = k
            mean_distortion_by_round_list = decode_mean_distortion(mean_distortion_by_round[last_k])
            katsavounidis_results[str(int(samples_random_seed))] = mean_distortion_by_round_list[-1] 

        if initial_alphabet_opt == 'xiaoxiao' and distortion_measure_opt == 'mse':
            last_k = ''
            for k in mean_distortion_by_round.keys():
                last_k = k
            mean_distortion_by_round_list = decode_mean_distortion(mean_distortion_by_round[last_k])
            xiaoxiao_results[str(int(samples_random_seed))] = mean_distortion_by_round_list[-1] 

        if initial_alphabet_opt == 'sa' and distortion_measure_opt == 'mse':
            last_k = ''
            for k in mean_distortion_by_round.keys():
                last_k = k
            mean_distortion_by_round_list = decode_mean_distortion(mean_distortion_by_round[last_k])
            sa_results[str(int(samples_random_seed))] = mean_distortion_by_round_list[-1] 


        if initial_alphabet_opt == 'unitary_until_num_of_elements' and distortion_measure_opt == 'mse':
            last_k = ''
            for k in mean_distortion_by_round.keys():
                last_k = k
            mean_distortion_by_round_list = decode_mean_distortion(mean_distortion_by_round[last_k])
            unitary_until_num_of_elements_results[str(int(samples_random_seed))] = mean_distortion_by_round_list[-1] 


        if initial_alphabet_opt == 'random_from_samples' and distortion_measure_opt == 'mse':
            last_k = ''
            for k in mean_distortion_by_round.keys():
                last_k = k
            mean_distortion_by_round_list = decode_mean_distortion(mean_distortion_by_round[last_k])
            random_from_samples_results[str(int(samples_random_seed))] = mean_distortion_by_round_list[-1] 


        occurences.append(1)

    print('occurences.len: ', len(occurences))
    all_results = []

    samples_random_seeds = samples_random_seeds.items()
    samples_random_seeds_k = np.array([str(k[0]) for k in sorted(samples_random_seeds)])

    labels = samples_random_seeds_k


    # ---------------------------katsavounivis--------------------------------------
    katsavounidis_results = katsavounidis_results.items()
    print ('len(katsavounidis_results): ', len(katsavounidis_results))
    katsavounidis_v = np.array([float(v[1]) for v in sorted(katsavounidis_results)])
    all_results.append(katsavounidis_v)


    #---------------------------xiaoxiao-----------------------------------------------
    xiaoxiao_results = xiaoxiao_results.items()
    print ('len(xiaoxio_results): ', len(xiaoxiao_results))
    xiaoxiao_v = np.array([float(v[1]) for v in sorted(xiaoxiao_results)])
    all_results.append(xiaoxiao_v)


    #---------------------------sa-----------------------------------------------
    sa_results = sa_results.items()
    print ('len(sa_results): ', len(sa_results))
    sa_v = np.array([float(v[1]) for v in sorted(sa_results)])
    all_results.append(sa_v)


    #---------------------------unitary_until_num_of_elements-----------------------------------------------
    unitary_until_num_of_elements_results = unitary_until_num_of_elements_results.items()
    print ('len(unitary_until_num_of_elements_results): ', len(unitary_until_num_of_elements_results))
    unitary_until_num_of_elements_v = np.array([float(v[1]) for v in sorted(unitary_until_num_of_elements_results)])
    all_results.append(unitary_until_num_of_elements_v)

    #---------------------------random_from_samples-----------------------------------------------
    random_from_samples_results = random_from_samples_results.items()
    print ('len(random_from_samples_results): ', len(random_from_samples_results))
    random_from_samples_v = np.array([float(v[1]) for v in sorted(random_from_samples_results)])
    all_results.append(random_from_samples_v)

    #-----------------------------all results ---------------------------------------------------------
    results = [] 
    for result_list in all_results:
        for v in result_list:
            results.append(v)

    results = np.array(results)

    results_mean = np.mean(results)
    results_stddev = np.sqrt(np.mean((results - results_mean) ** 2))

    intervallist = []
    conf_interval = []
    k = 2.06

    # ---------------------------katsavounivis--------------------------------------

    katsavounidis_mean = np.mean(katsavounidis_v)
    katsavounidis_var = np.mean((katsavounidis_v - katsavounidis_mean) ** 2)
    katsavounidis_stddev = np.sqrt(katsavounidis_var)
    katsavounidis_upbound = katsavounidis_mean + k * katsavounidis_stddev/np.sqrt(len(katsavounidis_results))
    katsavounidis_lowbound = katsavounidis_mean - k * katsavounidis_stddev/np.sqrt(len(katsavounidis_results))
    intervallist.append([katsavounidis_lowbound, katsavounidis_mean, katsavounidis_upbound])
    conf_interval.append([katsavounidis_lowbound, katsavounidis_upbound])

    #---------------------------xiaoxiao-----------------------------------------------

    xiaoxiao_mean = np.mean(xiaoxiao_v)
    xiaoxiao_var = np.mean((xiaoxiao_v - xiaoxiao_mean) ** 2)
    xiaoxiao_stddev = np.sqrt(xiaoxiao_var)
    xiaoxiao_upbound = xiaoxiao_mean + k * xiaoxiao_stddev/np.sqrt(len(xiaoxiao_results))
    xiaoxiao_lowbound = xiaoxiao_mean - k * xiaoxiao_stddev/np.sqrt(len(xiaoxiao_results))
    intervallist.append([xiaoxiao_lowbound, xiaoxiao_mean, xiaoxiao_upbound])
    conf_interval.append([xiaoxiao_lowbound, xiaoxiao_upbound])



    #---------------------------sa-----------------------------------------------

    sa_mean = np.mean(sa_v)
    sa_var = np.mean((sa_v - sa_mean) ** 2)
    sa_stddev = np.sqrt(sa_var)
    sa_upbound = sa_mean + k * sa_stddev/np.sqrt(len(sa_results))
    sa_lowbound = sa_mean - k * sa_stddev/np.sqrt(len(sa_results))
    intervallist.append([sa_lowbound, sa_mean, sa_upbound])
    conf_interval.append([sa_lowbound, sa_upbound])


    #---------------------------unitary_until_num_of_elements-----------------------------------------------

    unitary_until_num_of_elements_mean = np.mean(unitary_until_num_of_elements_v)
    unitary_until_num_of_elements_var = np.mean((unitary_until_num_of_elements_v - unitary_until_num_of_elements_mean) ** 2)
    unitary_until_num_of_elements_stddev = np.sqrt(unitary_until_num_of_elements_var)
    unitary_until_num_of_elements_upbound = unitary_until_num_of_elements_mean + k * unitary_until_num_of_elements_stddev/np.sqrt(len(unitary_until_num_of_elements_results))
    unitary_until_num_of_elements_lowbound = unitary_until_num_of_elements_mean - k * unitary_until_num_of_elements_stddev/np.sqrt(len(unitary_until_num_of_elements_results))
    intervallist.append([unitary_until_num_of_elements_lowbound, unitary_until_num_of_elements_mean, unitary_until_num_of_elements_upbound])
    conf_interval.append([unitary_until_num_of_elements_lowbound, unitary_until_num_of_elements_upbound])

    #---------------------------random_from_samples-----------------------------------------------

    random_from_samples_mean = np.mean(random_from_samples_v)
    random_from_samples_var = np.mean((random_from_samples_v - random_from_samples_mean) ** 2)
    random_from_samples_stddev = np.sqrt(random_from_samples_var)
    random_from_samples_upbound = random_from_samples_mean + k * random_from_samples_stddev/np.sqrt(len(random_from_samples_results))
    random_from_samples_lowbound = random_from_samples_mean - k * random_from_samples_stddev/np.sqrt(len(random_from_samples_results))
    intervallist.append([random_from_samples_lowbound, random_from_samples_mean, random_from_samples_upbound])
    conf_interval.append([random_from_samples_lowbound, random_from_samples_upbound])



    fig, ax = plt.subplots()

    #ax.boxplot((katsavounidis_v, xiaoxiao_v, sa_v, unitary_until_num_of_elements_v, random_from_samples_v), vert=True, showmeans=True, meanline=True,
    #       labels=('katsavounidis', 'xiaoxiao', 'sa', 'unitary', 'random'), patch_artist=True,
    #       medianprops={'linewidth': 2, 'color': 'purple'},
    #       meanprops={'linewidth': 2, 'color': 'red'})
    #plt.show()

    #plt.savefig('graph1.png')
    katsavounidis_1st_percentile = np.percentile(katsavounidis_v, 25) 
    katsavounidis_median = np.percentile(katsavounidis_v, 50) 
    katsavounidis_3rd_percentile = np.percentile(katsavounidis_v, 75) 
    katsavounidis_iqr = katsavounidis_3rd_percentile - katsavounidis_1st_percentile

    xiaoxiao_1st_percentile = np.percentile(xiaoxiao_v, 25) 
    xiaoxiao_median = np.percentile(xiaoxiao_v, 50) 
    xiaoxiao_3rd_percentile = np.percentile(xiaoxiao_v, 75) 
    xiaoxiao_iqr = xiaoxiao_3rd_percentile - xiaoxiao_1st_percentile

    sa_1st_percentile = np.percentile(sa_v, 25) 
    sa_median = np.percentile(sa_v, 50) 
    sa_3rd_percentile = np.percentile(sa_v, 75) 
    sa_iqr = sa_3rd_percentile - sa_1st_percentile

    unitary_until_num_of_elements_1st_percentile = np.percentile(unitary_until_num_of_elements_v, 25)
    unitary_until_num_of_elements_median = np.percentile(unitary_until_num_of_elements_v, 50)
    unitary_until_num_of_elements_3rd_percentile = np.percentile(unitary_until_num_of_elements_v, 75)
    unitary_until_num_of_elements_iqr = unitary_until_num_of_elements_3rd_percentile - unitary_until_num_of_elements_1st_percentile

    random_from_samples_1st_percentile = np.percentile(random_from_samples_v, 25)
    random_from_samples_median = np.percentile(random_from_samples_v, 50)
    random_from_samples_3rd_percentile = np.percentile(random_from_samples_v, 75)
    random_from_samples_iqr = random_from_samples_3rd_percentile - random_from_samples_1st_percentile


    whis_value = 1.57
    whiskers = [katsavounidis_1st_percentile - whis_value * katsavounidis_iqr, katsavounidis_3rd_percentile + whis_value * katsavounidis_iqr, 
                xiaoxiao_1st_percentile - whis_value * xiaoxiao_iqr, xiaoxiao_3rd_percentile + whis_value * xiaoxiao_iqr, 
                sa_1st_percentile - whis_value * sa_iqr, sa_3rd_percentile + whis_value * sa_iqr, 
                unitary_until_num_of_elements_1st_percentile - whis_value * unitary_until_num_of_elements_iqr, unitary_until_num_of_elements_3rd_percentile + whis_value * unitary_until_num_of_elements_iqr, 
                random_from_samples_1st_percentile - whis_value * random_from_samples_iqr, random_from_samples_3rd_percentile + whis_value * random_from_samples_iqr]


    bp_result = plt.boxplot((katsavounidis_v, xiaoxiao_v, sa_v, unitary_until_num_of_elements_v, random_from_samples_v), 
             #whis = whiskers,
             #whis = [katsavounidis_1st_percentile, katsavounidis_3rd_percentile, xiaoxiao_1st_percentile, xiaoxiao_3rd_percentile, sa_1st_percentile, sa_3rd_percentile, unitary_until_num_of_elements_1st_percentile, unitary_until_num_of_elements_3rd_percentile, random_from_samples_1st_percentile, random_from_samples_3rd_percentile],
             whis = whis_value,
             #sym='',
             #notch = True,
             #conf_intervals = np.array(conf_interval), 
             labels=('katsavounidis', 'xiaoxiao', 'sa', 'unitary', 'random'), 
             usermedians=[katsavounidis_median, xiaoxiao_median, sa_median, unitary_until_num_of_elements_median, random_from_samples_median],
             showmeans = True,
             meanline = True,
             patch_artist=True,
             medianprops={'linewidth': 2, 'color': 'purple'},
             meanprops={'linewidth': 2, 'color': 'red'})
    plt.show()
    #plt.savefig('graph2.png')
    for k, v in bp_result.items():
        print ('k: ', k)

    whiskers = bp_result['whiskers']
    caps = bp_result['caps']
    boxes = bp_result['boxes']
    medians = bp_result['medians']
    fliers = bp_result['fliers']
    means = bp_result['means']
    

    print ('means')
    for d in means:
        print (d.get_data())
    print ('----------------')




    print ('fliers')
    for d in fliers:
        print (d.get_data())
    print ('----------------')


    print ('medians')
    for d in medians:
        print (d.get_data())
    print ('----------------')

    #print ('boxes')
    #for d in boxes:
    #    print (d.get_data())
    #print ('----------------')


    print ('caps')
    for d in caps:
        print (d.get_data())
    print ('----------------')
 


    print ('whiskers')
    for d in whiskers:
        print (d.get_data())
    print ('----------------')
    
    #usermedians=[katsavounidis_median, xiaoxiao_median, sa_median, unitary_until_num_of_elements_median, random_from_samples_median],
    print (conf_interval)


    
#    intervallist.append([random_from_samples_lowbound, random_from_samples_mean, random_from_samples_upbound])
#
#    #x = np.arange(len(labels))
#    x = np.arange(1)
#    width = 0.001
#    
#    fig, ax = plt.subplots()
#
#    #rects1 = ax.bar(x + width, katsavounidis_mean, width, label='katsavounidis', yerr=katsavounidis_stddev)
#    #rects2 = ax.bar(x + 2 * width, xiaoxiao_mean, width, label='xiaoxiao', yerr=xiaoxiao_stddev)
#    #rects3 = ax.bar(x + 3 * width, sa_mean, width, label='sa', yerr=sa_stddev)
#    #rects4 = ax.bar(x + 4 * width, unitary_until_num_of_elements_mean, width, label='unitary', yerr=unitary_until_num_of_elements_stddev)
#    #rects5 = ax.bar(x + 5 * width, random_from_samples_mean, width, label='random', yerr=random_from_samples_stddev)
#
#    plt.boxplot(intervallist)
#
#    ax.set_ylabel('Minimal distortion')
#    ax.set_xlabel('Seed samples from trials')
#    ax.set_title('Minimal distortion by initial alphabet method - Nt = 16, k = 8000, var = 1.0')
#    ax.set_xticks(x)
#    #ax.set_xticklabels(labels)
#    ax.legend()
#
#    plt.show()
#
#    #print (sorted(random_from_samples_results))
#
#    ##norm_values_array_l1 = np.array(sorted(norm_values_l1, key=lambda k: k['norm'], reverse=True))
#    #norm_values_array_l1 = np.array([v['norm'] for v in norm_values_l1])
#    #norm_values_array_l1 = norm_values_array_l1/np.sqrt((np.sum(np.power(norm_values_array_l1, 2))))
#    #plt.plot(norm_values_array_l1, 'r*', label='variance = 0.1')
#
#    ##norm_values_array_l2 = np.array(sorted(norm_values_l2, key=lambda k: k['norm'], reverse=True))
#    #norm_values_array_l2 = np.array([v['norm'] for v in norm_values_l2])
#    #norm_values_array_l2 = norm_values_array_l2/np.sqrt((np.sum(np.power(norm_values_array_l2, 2))))
#    #plt.plot(xiaoxiao_v, '-', label='xiaoxiao_v')
#
#
#
#    #plt.savefig('results_graph1.png')
