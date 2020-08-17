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
    count = 0
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

        normal_vector = np.ones(num_of_levels) * (num_of_samples/num_of_levels)

        if initial_alphabet_opt == 'random_from_samples' and num_of_levels == 4 and variance_of_samples == 0.01 and distortion_measure_opt == 'mse':
            sets = d['sets']
            set_vector = []
            for k, v in sets.items():
                set_vector.append(v)
            set_vector = np.array(set_vector)
   
            norm =  np.sqrt(np.sum(np.power(np.abs(set_vector - normal_vector), 2)))
            print ('norm: ', norm)
            if norm == 0.0:
                print ('trial file results: ', pathfile)
                print ('norm: ', norm)
                count += 1
            #print (num_of_levels)
            #count += 1

    print ('total # of normal distribution results:', count)
