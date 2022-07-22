import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict 
import pickle


def count_labels():
    lbl_counts = {0:{}, 1:{}, 2:{}}
    dataPath = 'input/raw'
    envlist = sorted(os.listdir(dataPath))
    for category in ['train-labels_new']:
        path = os.path.join(dataPath, category)
        samplefiles = sorted(os.listdir(path))
        for sample in samplefiles:
            print("processing sample: train-labels/{sample}".format(sample=sample))
            lblfilePath = os.path.join(path, sample)
            lbl = np.load(lblfilePath)

            for i in range(lbl.shape[2]):
                lbl_vals, counts_ = np.unique(lbl[:,:,i], return_counts=True)
                for j, lbl_val in enumerate(lbl_vals):
                    if lbl_val not in lbl_counts[i].keys():
                        lbl_counts[i][lbl_val] = 0
                    lbl_counts[i][lbl_val] += counts_[j]
    
    pickle.dump(lbl_counts, open('results/lbl_dist.pkl', 'wb'))
    return lbl_counts


def plot_dist(lbl_counts):
    print(lbl_counts)
    dof1_counts = []
    for i in range(8):
        if i not in lbl_counts[1].keys():
            dof1_counts.append(0)
        else:
            dof1_counts.append(lbl_counts[1][i]/5000)
    
    plt.bar(np.arange(8), height=dof1_counts)
    plt.show()

    dof2_counts = []
    for i in range(9):
        if i not in lbl_counts[2].keys():
            dof2_counts.append(0)
        else:
            dof2_counts.append(lbl_counts[2][i]/5000)
    
    plt.bar(np.arange(9), height=dof2_counts)
    plt.show()

if __name__ == "__main__":
    
    lbl_counts = count_labels()
    # lbl_counts = pickle.load(open('results/lbl_dist.pkl', 'rb'))
    # plot_dist(lbl_counts)