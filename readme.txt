Envs, Model, and data are available to download from the conference chairs. 

Dependencies: 
python2 
OpenRave 0.9.0
Tensorflow 1.14.0
Numpy 
Scipy
opencv
networkx 


To run the approach: 

1) Place corresponding envs in corresponding folder
2) Place learned model in network folder 
3) Geneate Input Image using generate_input_image.py (for small envs) and generate_input_image_multi_input.py (for large envs - env41.0, env42.0, env43.0)
4) Generate predictions for the current environment using test.py in network folder. You need to generate tfrecords to do so. For large environments each input image has to be used separately. 
5) Run HARP.py (small envs) and HARP_Large.py (large envs).  
