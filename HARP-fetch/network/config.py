model = 'unetv2'  # which model to use. load in utils.py
working_dataset = 'fetch-v1'  # dataset name in input/ folder 
gpu_memory_fraction = 0.99  # Limit GPU usage
strided = False  # Use strided inference or not (not used. kept for backward compatibility)

number_of_dof = 8  # number of degrees of freedom of robot
number_of_bins = 10  # number of bins each DOF is divided into
input_channels = 11  # number of channels in network input
output_channels = 81  # number of channels in network output
input_size = 64  # size of input
output_size = 224  # size of output
dof1_bins = 10
# input_shape = (input_size, input_size, input_size, input_channels)  # input shape
input_shape = (input_size,input_size,input_size,input_channels)
# label_shape = (output_size, output_size, output_size, output_channels)  # output shape

label_shape = (input_size,input_size,input_size, 9)
# label_shape = (number_of_dof-1,number_of_bins,number_of_bins)