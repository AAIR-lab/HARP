import numpy as np
import sys

pd = np.load(sys.argv[1])
if len(pd.shape) > 3:
    pd = pd.reshape(pd.shape[-3],pd.shape[-2],pd.shape[-1])

xy_pd = pd[0,:,:]

np.savetxt(sys.argv[2],xy_pd,delimiter=",")