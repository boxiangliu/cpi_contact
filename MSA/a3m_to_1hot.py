'''
INPUT:
	alignment file in a3m format
OUTPUT:
	MSA in one-hot format
'''

import sys
import torch
import torch.nn.functional as F
import numpy as np
import string

a3m_fn = sys.argv[1]
onehot_fn = sys.argv[2]

# read A3M and convert letters into
# integers in the 0..20 range
def parse_a3m(filename):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file line by line
    for line in open(filename,"r"):
        # skip labels
        if line[0] != '>':
            # remove lowercase letters and right whitespaces
            seqs.append(line.rstrip().translate(table))

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    return msa

msa = parse_a3m(a3m_fn)
onehot = F.one_hot(torch.LongTensor(msa), 21)

torch.save(onehot, onehot_fn)

