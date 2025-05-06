import os
import shutil
path = "./data/"
op = "./su3/"
for i in os.listdir(op):
    dir = op+i+"/"
    for k in os.listdir(dir):
        shutil.move(dir+k, path+i+"_"+k)