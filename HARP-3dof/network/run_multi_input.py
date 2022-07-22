import os
import sys
import time

# env = sys.argv[1]
env = "env43.0"
fname_base = env[3:] + ".1"


for i in range(5):
    for j in range(5):
        os.system("rm -rf ./input/raw/test/*")
        fname = fname_base + "_" + str(i) + "_" +str(j) + ".npy"
        os.system("cp ../test/{}/{} ./input/raw/test/".format(env,fname))
        os.system("./test.sh {}".format(fname[:-4]))
        time.sleep(2)

