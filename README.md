# Vector Quantization applied on Codebook Design problem using Generalized Lloyd Algorithm (GLA)

## 1. Main reference
*[1] Linde, Y., Buzo, A., & Gray, R. (1980). An algorithm for vector quantizer design. IEEE Transactions on communications, 28(1), 84-95.*

## 2. Problem description

This code begins defining a DFT codebook with an arbitrary size Nt. This codebook is an NtxNt matrix of complex-valued. In the codebook matrix, every single line (1xNt) is a codeword who is orthogonal to each other. Then, n noisily contaminated training samples are created from DFT codebook. With the same shape of the codewords, a sample is a 1xNt complex-valued vector. Finally, these n samples are presented to the Lloyd LBG algorithm who returns a quantizer of L=Nt levels, a histogram with the number of samples grouped by codeword, and the perform of distortion along with training processes. In this toy problem, we consider as the ideal quantizer (who produces the optimum distortion) that DFT codebook originally used to produce the training samples.


## 3. Implementation
As stated in [1], the algorithm produces a quantizer meeting necessary but not sufficient conditions for optimality. Usually, however, at least local optimality is assured, and the choice of initial reconstruct alphabet looks like to be crucial to define a better quantizer. There are several ways to chose it. Trying to get the best quantizer (something similar to the original DFT codebook), this code implements two ways to choice initial reconstruct alphabet:

* (I) selecting *L* random codewords from samples; and, 
* (II) considering initial reconstruct alphabet as M-level quantizers with *M=2^R*, for *R* in *[0, 1, 2,..., log2(L)]*. 

Furthermore, two distortion measures options can be used in each blend of GLA algorithm: 

* (1) Squared Error, and 
* (2) Gain as the internal product. 

The results are encoded and stored in JSON data files.

## 4. Download this code
* $ git clone https://github.com/sergiossc/lloyd-gla

## 5. Required additional python packages
* This code was made in python3.6.
* We need of the following additional packages: matplotlib>=3.1.3; and, numpy>=1.18.1

To install these additional packages try:
* $ python3.6 -m pip install -r requirements.txt


## 6. Running an example
1. Edit the 'profile.json' file and set the general parameters like this:


```

    "number_of_elements": [4, 16], # Nt values 
    
    "variance_of_samples_values": [0.01, 0.1], # Values of the variance of samples
    
    "initial_alphabet_opts": ["random_from_samples", "unitary_until_num_of_elements"], # Options of initial alphabet
    
    "distortion_measure_opts": ["mse", "gain"], # Distortion measure  options
    
    "num_of_trials": 1, # Number of trials to run to a given setup
    
    "num_of_samples": 80, # Number of training samples
    
    "num_of_interactions": 2, # Number max of interactions during training. 20 is a good number.
    
    "results_directory": "/home/path/to/lloyd-lbg/results", # Path to save result JSON files of each trial
    
    "use_same_samples_for_all": true # Set it true (or false) if you want to use (or not) the same training samples over all trials
```

* Considering the information in 'profile.json' file, the total number of trials is *[len(number_of_elements) x len(variance_of_samples_values) x len(distortion_measure_opts) x num_of_trials]*. Running the code with this 'profile.json' above we should get [2 x 2 x 2 x 2 x 1] = 16 trials and their corresponding JSON result file.

2. Running the code

Use 'lloyd.py' script to run the code:
*  $ python3.6 lloyd.py

In this code, each trial is a python process.

3. Reading JSON results files

To read json result files use *identifying_normal_results.py* script
*  $ python3.6 identifying\_normal\_results.py

4. Comparing final reconstruct alphabet with original DFT codebook

Use *comparing_codewords.py* script to compares a specific final estimated codebook with the original DFT codebook

*  $ python3.6 comparing\_codewords.py <trial\_results\_jsonfile.json>
