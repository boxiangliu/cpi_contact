import yaml
import sys
import os
import urllib
from tqdm import tqdm
from collections import defaultdict

config_fn = sys.argv[1]
config_fn = "dataset/config.yaml"


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


class DataDownloader(DataUtils):

    def __init__(self, config_fn, debug=False):
        super(PdbDownloader, self).__init__(config_fn)
        self.debug = debug
        self.pdb_url = "https://files.rcsb.org/download/"
        self.ligand_url = "https://files.rcsb.org/ligands/download/"
        self.uniprot_url = "https://www.uniprot.org/uploadlists/"

        self.pdbid_list = self.get_pdbid_list(self.config)
        self.download_complex(
            self.pdbid_list, self.pdb_url, self.config)

        self.pdbid_to_ligand = self.get_pdb_to_ligand(self.config)
        self.download_ligand(
            self.pdbid_list, self.pdbid_to_ligand, self.ligand_url, self.config)

        self.pdb_to_uniprot = self.download_pdb_to_uniprot(
            self.pdbid_list, self.uniprot_url)
        self.uniprot_id_set = self.get_uniprot_id_set(
            self.config, self.pdb_to_uniprot)

        self.fasta = self.download_fasta(
            self.config, self.uniprot_id_set, self.uniprot_url)

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

        out_dir = config["DATA"]["PDB"]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        n = 0
        for pdbid in tqdm(pdbid_list):
            url = url_prefix + pdbid + '.pdb'
            out_fn = os.path.join(out_dir, pdbid + '.pdb')
            if not os.path.exists(out_fn):
                n += 1
                urllib.request.urlretrieve(url, out_fn)

        sys.stderr.write(f"Number of downloaded complexes: {n}\n")

    def get_pdb_to_ligand(self, config):
        pdbid_to_ligand = {}
        index_dir = config["PDBBIND"]["INDEX"]
        index_fn = os.path.join(index_dir, "INDEX_general_PL.2018")
        with open(index_fn, "r") as f:
            for line in f:
                if line[0] != '#':
                    ligand = line.strip().split('(')[1].split(')')[0]
                    if '-mer' in ligand:
                        continue
                    elif '/' in ligand:
                        ligand = ligand.split('/')[0]
                    if len(ligand) != 3:
                        continue
                    pdbid_to_ligand[line[:4]] = ligand
        sys.stderr.write(f'Number of ligand: {len(pdbid_to_ligand)}\n')
        return pdbid_to_ligand

    def download_ligand(self, pdbid_list, pdbid_to_ligand, url_prefix, config):
        out_dir = config["DATA"]["PDB"]

        if self.debug:
            pdbid_list = pdbid_list[:5]

        n = 0
        for pdbid in tqdm(pdbid_list):
            if pdbid in pdbid_to_ligand:
                ligand = pdbid_to_ligand[pdbid]
                url = url_prefix + ligand + "_ideal.pdb"
                out_fn = os.path.join(out_dir, ligand + '_ideal.pdb')
                if not os.path.exists(out_fn):
                    n += 1
                    urllib.request.urlretrieve(url, out_fn)

        sys.stderr.write(f"Number of downloaded complexes: {n}\n")

    def download_pdb_to_uniprot(self, pdbid_list, uniprot_url):
        query = " ".join(pdbid_list)
        params = {
            'from': 'PDB_ID',
            'to': 'ACC',
            'format': 'tab',
            'query': query
        }

        response = self.query_uniprot(params, uniprot_url)

        lines = response.split("\n")
        mapping = defaultdict(list)
        for line in lines:
            if len(line) > 0 and not line.startswith("From"):
                pdb_id, uniprot_id = line.split("\t")
                mapping[pdb_id].append(uniprot_id)

        return mapping

    def get_uniprot_id_set(self, config, pdb_to_uniprot):
        uniprotid_set = set()
        index_dir = config["PDBBIND"]["INDEX"]
        index_fn = os.path.join(index_dir, "INDEX_general_PL_name.2018")
        with open(index_fn, "r") as f:
            for line in f.readlines():
                if line[0] != '#':
                    lines = line.strip().split('  ')
                    if lines[2] != '------':
                        uniprotid_set.add(lines[2])
        sys.stderr.write(f'Number of UniProt IDs from PDBbind: {len(uniprotid_set)}\n')

        uniprotid_set_2 = []
        for uid in pdb_to_uniprot.values():
            uniprotid_set_2.extend(uid)

        uniprotid_set_2 = set(uniprotid_set_2)
        sys.stderr.write(f"Number of UniProt IDs from website: {len(uniprotid_set_2)}\n")

        uniprotid_set = uniprotid_set.union(uniprotid_set_2)
        sys.stderr.write(f"Total number of unique UniProt IDs: {len(uniprotid_set)}\n")

        return uniprotid_set

    def download_fasta(self, config, uniprot_id_set, uniprot_url):
        query = " ".join(list(uniprot_id_set))
        params = {
            'from': 'ACC',
            'to': 'ACC',
            'format': 'fasta',
            'query': query
        }

        response = self.query_uniprot(params, uniprot_url)

        out_dir = config["DATA"]["WD"]
        out_fn = os.path.join(out_dir, "out1.6_pdbbind_seqs.fasta")
        with open(out_fn, "w") as f:
            f.write(response)

data_downloader = DataDownloader(config_fn, debug=True)


class DataProcessor(DataUtils):

    def __init__(self, config_fn, debug=False):
        super(DataProcessor, self).__init__(config_fn)
        self.debug = debug

        self.mol_dict = self.get_mol_dict(self.config)

    def get_mol_dict(self, config):
        coordinate_fn = config["PDB"]["COORDINATE"]
        mol_dict = {}
        mols = Chem.SDMolSupplier(coordinate_fn)
        for m in mols:
            if m is None:
                continue
            name = m.GetProp("_Name")
            mol_dict[name] = m
        sys.stderr.write(f'Number of Molecules in Components-pub.sdf: {len(mol_dict)}\n')
        return mol_dict

data_processor = DataProcessor(config_fn)