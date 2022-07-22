import os

DATADIR = "./data-v2/"
TARGET = "train"

dir_list = os.listdir(DATADIR)
for d in dir_list:
    os.system("cp {}/{}/inp/* ./network/input/raw/{}/".format(DATADIR,d,TARGET))
    os.system("cp {}/{}/lbl/* ./network/input/raw/{}-labels/".format(DATADIR,d,TARGET))

