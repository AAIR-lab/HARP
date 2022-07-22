import os
import sys
import subprocess

base = "./data/"
env_base = "./envs3d/"
files = os.listdir(base)
env_files = os.listdir(env_base)
paths = {}
for f in files:
    f = f[3:]
    if "{}.xml".format(f) in env_files:
        paths[f] = "{}{}.xml".format(env_base,f)
    elif "{}.dae".format(f) in env_files:
        paths[f] = "{}{}.dae".format(env_base,f)

for name in paths:
    if "env8.0" not in name and "env10.0" not in name:
        for run in range(1,101,1):
            cmd = ["python generate_multinetwork_labels.py {} {} {}".format(paths[name],name,run)]
            # cmd = ["python generate_labelsV3.py {} {} {}".format(paths[name],name,run)]
            p = subprocess.Popen(cmd,shell=True)
            p.wait()
            # subprocess.call(cmd,shell=True)