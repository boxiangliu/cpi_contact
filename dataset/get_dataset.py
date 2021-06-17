import yaml
import sys
import os
from urllib.request import urlretrieve
from tqdm import tqdm

config_fn = sys.argv[1]
config_fn = "dataset/config.yaml"


class DataDownloader():

    def __init__(self, config_fn):
        self.config_fn = config_fn
        self.config = self.parse_config(config_fn)

    def parse_config(self, config_fn):
        with open(config_fn, "r") as f:
            config = yaml.safe_load(f)
        return config


class PdbDownloader(DataDownloader):

    def __init__(self, config_fn, debug=False):
        super(PdbDownloader, self).__init__(config_fn)
        self.debug = debug
        self.pdbid_list = self.get_pdbid_list(self.config)
        self.pdb_url_prefix = "https://files.rcsb.org/download/"
        self.download_complex(self.pdbid_list, self.pdb_url_prefix, self.config)

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

    def download_complex(self, pdbid_list, url_prefix, config):
        if self.debug:
            pdbid_list = pdbid_list[:5]

        n = 0
        for pdbid in tqdm(pdbid_list):
            n += 1
            url = url_prefix + pdbid + '.pdb'
            out_dir = config["DATA"]["PDB"]
            out_fn = os.path.join(out_dir, pdbid + '.pdb')
            urlretrieve(url, out_fn)
        sys.stderr.write(f"Number of downloaded complexes: {n}\n")
