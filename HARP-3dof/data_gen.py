import subprocess
import sys

envlist = sys.argv[1:]

for e in envlist:
    for i in range(1,101,1):
        cmd = ["python generate_mp.py {} {}".format(str(e),str(i))]
        p = subprocess.Popen(cmd,shell=True)
        p.wait()
