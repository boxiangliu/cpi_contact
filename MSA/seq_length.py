import os
import glob
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt 

msa_dir = "/mnt/scratch/boxiang/projects/cpi_contact/data/MSA"

seq_len = []
for fn in tqdm(glob.glob(os.path.join(msa_dir, "*.a3m"))):
    with open(fn) as f:
        f.readline()
        seq = f.readline()
        seq_len.append(len(seq))

bool_ = [int(x > 1024) for x in seq_len]
sum(bool_) / len(bool_)
# 0.1514

plt.hist(seq_len)