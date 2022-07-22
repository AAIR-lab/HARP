import numpy as np
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt

'''
view the input image and the labels plotted on a 6x7 graph
'''

envnum = 17.2
runnum = 1
suffixes = ['', 'f1', 'r1', 'r2', 'r3']
inpfiles = []
lblfiles = []
for suffix in suffixes:
    inpfiles.append('data/env'+str(envnum)+'/inp/'+str(envnum)+'.'+str(runnum)+ suffix + '.npy')
    lblfiles.append('data/env'+str(envnum)+'/lbl/'+str(envnum)+'.'+str(runnum)+ suffix + '.npy')
print(inpfiles)
print(lblfiles)

num_rows = len(suffixes)
num_columns = 1 + 3 # 1 input env + 3 label channels
f, axarr = plt.subplots(num_rows, num_columns)
k = 0

# iterate every augmented type
# load its plots in a row
for i in xrange(len(inpfiles)):

    inp = np.load(inpfiles[i])
    lbl = np.load(lblfiles[i])

    axarr[i][0].imshow(inp[:,:,:3])
    for j in range(1, num_columns):
        axarr[i][j].imshow(lbl[:,:,j-1])

y_labels = ['original', 'f1', 'r1', 'r2', 'r3']
j = 0
for i, ax in enumerate(axarr.flat):
    if i%num_columns == 0:
        j+=1
    ax.set(ylabel=y_labels[j-1])
    ax.label_outer()
plt.show()