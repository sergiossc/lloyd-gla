
executable = run_lloyd.py
log = run_lloyd.$(Cluster).$(Process).out
error = run_lloyd.$(Cluster).$(Process).err
should_transfer_files = Yes
transfer_input_files = utils.py
when_to_transfer_output = ON_EXIT
arguments = 16 1.0 katsavounidis mse 80 1000 /home/users/sergiossc/current/lloyd-gla/results 986ba16a-e417-486e-95f8-3cecefc88af9 1 77697 764 
queue
arguments = 16 1.0 xiaoxiao mse 80 1000 /home/users/sergiossc/current/lloyd-gla/results aff5f3d2-5939-412f-8c39-b6b8a7945188 1 77697 7314 
queue
arguments = 16 1.0 sa mse 80 1000 /home/users/sergiossc/current/lloyd-gla/results 303d1d56-4e5c-4005-9b63-898350a02231 1 77697 3216 
queue
arguments = 16 1.0 unitary_until_num_of_elements mse 80 1000 /home/users/sergiossc/current/lloyd-gla/results 79368c13-f6da-4a4d-9689-99f3c38ac6fe 1 77697 5991 
queue
arguments = 16 1.0 random_from_samples mse 80 1000 /home/users/sergiossc/current/lloyd-gla/results 423d7871-1de7-42d8-bffd-ba5c760cbb41 1 77697 537 
queue
