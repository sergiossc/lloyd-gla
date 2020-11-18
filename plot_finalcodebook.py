import json
import numpy as np
import sys
from utils import *

if __name__ == '__main__':

    pathfile = sys.argv[1]
        
    with open(pathfile) as result:
        data = result.read()
        d = json.loads(data)

    lloydcodebook_dict = decode_codebook(d['lloydcodebook'])

    lloydcodebook_matrix = dict2matrix(lloydcodebook_dict)
    print ('before:\n', lloydcodebook_matrix)
    nrows, ncols = lloydcodebook_matrix.shape
    lloydcodebook_matrix = np.array(lloydcodebook_matrix).reshape(nrows, 1, ncols)
    print ('after:\n', lloydcodebook_matrix)
    plot_codebook(lloydcodebook_matrix, 'final_codebook.png')

        


