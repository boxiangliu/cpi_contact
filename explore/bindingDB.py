import pandas as pd
import numpy as np
from collections import defaultdict

bindingDB_file = "../data/bindingDB/BindingDB_All.tsv"


def read_bindingDB(bindingDB_file):
    '''
    Function to read the first 49 columns of bindingDB
    into a container. 

    Note: pandas fails to read the bindingDB file. 
    '''
    with open(bindingDB_file, "r") as f:
        container = defaultdict(list)
        for i, line in enumerate(f):
            split_line = line.split("\t")[:49]
            if i == 0:
                keys = split_line
            else:
                for key, value in zip(keys, split_line):
                    container[key].append(value.strip())
        return container

bindingDB = read_bindingDB(bindingDB_file)
bindingDB = pd.DataFrame(bindingDB)

bindingDB[(bindingDB["Ki (nM)"] != "") & 
            (bindingDB["Ligand HET ID in PDB"] != "") & 
            (bindingDB["PDB ID(s) for Ligand-Target Complex"] != "")]\
            [["Ki (nM)", "Ligand HET ID in PDB", "PDB ID(s) for Ligand-Target Complex"]]