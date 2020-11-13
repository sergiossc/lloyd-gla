import uuid
import json
import numpy as np
import os

def check_files(prefix, episodefiles):
    pathfiles = {}
    for ep_file in episodefiles:
        pathfile = prefix + str('/') + str(ep_file)
        ep_file_status = False
        try:
            current_file = open(pathfile)
            ep_file_status = True
            #print("Sucess.")
        except IOError:
            print("File not accessible: ", pathfile)
        finally:
            current_file.close()

        if ep_file_status:
            ep_file_id = uuid.uuid4()
            pathfiles[ep_file_id] = pathfile
 
    return pathfiles


def decode_mean_distortion(mean_distortion_dict):
    mean_distortion_list = []
    for iteration, mean_distortion in mean_distortion_dict.items():
        mean_distortion_list.append(mean_distortion)
    return mean_distortion_list


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
    filters = [] 
    filter4 = {'initial_alphabet_opt': 'unitary_until_num_of_elements', 'distortion_measure_opt': 'gain', 'num_of_levels': 4, 'variance_of_samples': 0.25}
    #trial_result = (initial_alphabet_opt, distortion_measure_opt, num_of_levels, variance_of_samples, norm)
    filters.append(filter4)
    occurences = [] # it is an histogram of filters
    for pathfile_id, pathfile in pathfiles.items():
        with open(pathfile) as result:
            data = result.read()
            d = json.loads(data)

        # May you edit from right here! Tip: Read *json file in results to decide how to deal from here.
        initial_alphabet_opt = d['initial_alphabet_opt']
        num_of_levels = d['num_of_levels']
        num_of_samples = d['num_of_samples']
        variance_of_samples = d['variance_of_samples']
        distortion_measure_opt = d['distortion_measure_opt']
        use_same_samples_for_all = d['use_same_samples_for_all'] 
        initial_alphabet_opt = d['initial_alphabet_opt']
        initial_alphabet_method = d['initial_alphabet_method']
        num_of_elements = d['num_of_elements']

        normal_vector = np.ones(num_of_levels) * (num_of_samples/num_of_levels)
        sets = d['sets']
        set_vector = []
        for k, v in sets.items():
            set_vector.append(v)
        set_vector = np.array(set_vector)
   
        norm =  np.sqrt(np.sum(np.power(np.abs(set_vector - normal_vector), 2)))
        #if norm == 0 and num_of_elements == 9 and variance_of_samples == 1.0 and initial_alphabet_method == 'katsavounidis': 
        #if  norm == 0 and num_of_elements == 4 and variance_of_samples == 0.1 and initial_alphabet_method == 'katsavounidis'
        if  norm == 0 and num_of_elements == 9 and initial_alphabet_method == 'katsavounidis':
            occurences.append(1)
            print (pathfile)

        #trial_result = (initial_alphabet_opt, distortion_measure_opt, num_of_levels, variance_of_samples, norm)
        #trial_result = [(initial_alphabet_opt, distortion_measure_opt, num_of_levels, variance_of_samples, norm)]
        #trial_result = {'initial_alphabet_opt': initial_alphabet_opt, 'distortion_measure_opt': distortion_measure_opt, 'num_of_levels':num_of_levels, 'variance_of_samples':variance_of_samples}
        #occurences = filter(filtering, trial_result)
        #for f in filters:
        #    f_values = list(f.values())
        #    trial_values = list(trial_result.values())
        #    if f_values == trial_values:
        #        print (f)
        #        print (trial_result)
        #        occurences.append(trial_result)
    #for occurence in occurences:
    print(len(occurences))
