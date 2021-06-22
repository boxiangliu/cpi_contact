import yaml
import urllib.parse
import urllib.request
import os
import sys

class DataUtils():

    def __init__(self, config_fn):
        self.config_fn = config_fn
        self.config = self.parse_config(config_fn)

    def parse_config(self, config_fn):
        with open(config_fn, "r") as f:
            config = yaml.safe_load(f)
        return config

    def query_uniprot(self, params, uniprot_url):
        data = urllib.parse.urlencode(params)
        data = data.encode('utf-8')
        req = urllib.request.Request(uniprot_url, data)
        with urllib.request.urlopen(req) as f:
            response = f.read().decode("utf-8")

        return response

    def read_INDEX_general_PL_name(self, config):
        uniprotid_set = set()
        pdbid_set = set()
        pdbbind_mapping_dict = dict()

        index_dir = config["PDBBIND"]["INDEX"]
        index_fn = os.path.join(index_dir, "INDEX_general_PL_name.2018")
        with open(index_fn, "r") as f:
            for line in f.readlines():
                if line[0] != '#':
                    lines = line.strip().split('  ')
                    if lines[2] != '------':
                        pdbid_set.add(lines[0])
                        uniprotid_set.add(lines[2])
                        pdbbind_mapping_dict[lines[0]] = lines[2]
        return pdbid_set, uniprotid_set, pdbbind_mapping_dict

    def get_pdbid_list(self, config):
        pdbid_list = []
        index_dir = config["PDBBIND"]["INDEX"]
        index_fn = os.path.join(index_dir, "INDEX_general_PL.2018")
        with open(index_fn, "r") as f:
            for line in f:
                if line[0] != '#':
                    pdbid = line.strip().split()[0]
                    pdbid_list.append(pdbid)

        sys.stderr.write(f'Number of PDB IDs: {len(pdbid_list)}\n')

        return pdbid_list

    def get_pdb_to_ligand(self, config):
        pdbid_to_ligand = {}
        index_dir = config["PDBBIND"]["INDEX"]
        index_fn = os.path.join(index_dir, "INDEX_general_PL.2018")
        with open(index_fn, "r") as f:
            count_error = 0
            for line in f:
                if line[0] != '#':
                    ligand = line.strip().split('(')[1].split(')')[0]
                    if '-mer' in ligand:
                        continue
                    elif '/' in ligand:
                        ligand = ligand.split('/')[0]
                    if len(ligand) != 3:
                        count_error += 1
                        continue
                    pdbid_to_ligand[line[:4]] = ligand
        sys.stderr.write(f'Number of ligand: {len(pdbid_to_ligand)}\n')
        sys.stderr.write(f"Number of PDB ID without ligand: {count_error}\n")
        return pdbid_to_ligand
