import yaml
import sys
import os
import urllib.parse
import urllib.request
from tqdm import tqdm
from collections import defaultdict
from rdkit import Chem
import numpy as np
from utils import DataUtils

config_fn = "dataset/config.yaml"

class DataDownloader(DataUtils):

    def __init__(self, config_fn, debug=False, download=True, process=True):
        super(DataDownloader, self).__init__(config_fn)
        self.debug = debug
        self.pdb_url = "https://files.rcsb.org/download/"
        self.ligand_url = "https://files.rcsb.org/ligands/download/"
        self.uniprot_url = "https://www.uniprot.org/uploadlists/"

        if download:
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

        if process:
            self.mol_dict = self.get_mol_dict(self.config)
            self.uniprot_dict = self.get_fasta_dict(self.config)
            self.pdbid_to_uniprotid = self.get_pdbid_to_uniprotid(
                self.uniprot_dict)
            self.pdbid_to_measure, self.pdbid_to_value = self.get_pdbid_to_affinity(
                self.config)
            self.write_all_datafile(self.config, self.pdbid_to_uniprotid, self.pdbid_to_ligand,
                                    self.pdbid_to_measure, self.pdbid_to_value, self.mol_dict, self.uniprot_dict)

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

    def download_ligand(self, pdbid_list, pdbid_to_ligand, url_prefix, config):
        out_dir = config["DATA"]["PDB"]

        if self.debug:
            pdbid_list = pdbid_list[:5]

        n = 0
        error = 0
        for pdbid in tqdm(pdbid_list):
            if pdbid in pdbid_to_ligand:
                ligand = pdbid_to_ligand[pdbid]
                url = url_prefix + ligand + "_ideal.pdb"
                out_fn = os.path.join(out_dir, ligand + '_ideal.pdb')
                if not os.path.exists(out_fn):
                    try:
                        urllib.request.urlretrieve(url, out_fn)
                        n += 1
                    except:
                        error += 1

        sys.stderr.write(f"Number of downloaded ligands: {n}\n")
        sys.stderr.write(f"Number of missing ligands: {error}\n")

    def download_pdb_to_uniprot(self, pdbid_list, uniprot_url):
        '''
        This function returns the equivalent of out1.4_pdb_uniprot_mapping.tab 
        '''
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
        _, uniprotid_set, _ = self.read_INDEX_general_PL_name(config)
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
        out_dir = config["DATA"]["WD"]

        query = " ".join(list(uniprot_id_set))
        params = {
            'from': 'ACC',
            'to': 'ACC',
            'format': 'fasta',
            'query': query
        }
        response = self.query_uniprot(params, uniprot_url)
        out_fn = os.path.join(out_dir, "out1.6_pdbbind_seqs.fasta")
        with open(out_fn, "w") as f:
            f.write(response)

        params = {
            'from': 'ACC',
            'to': 'ACC',
            'format': 'tab',
            'query': query
        }
        response = self.query_uniprot(params, uniprot_url)
        out_fn = os.path.join(out_dir, "out1.6_uniprot_uniprot_mapping.tab")
        with open(out_fn, "w") as f:
            f.write(response)

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

    def get_fasta_dict(self, config):
        fasta_dict = {}

        name, seq = '', ''
        out_dir = config["DATA"]["WD"]
        fasta_fn = os.path.join(out_dir, 'out1.6_pdbbind_seqs.fasta')
        with open(fasta_fn, "r") as f:
            for line in f.readlines():
                if line[0] == '>':
                    if name != '':
                        fasta_dict[name] = seq
                    name = line.split('|')[1]
                    seq = ''
                else:
                    seq += line.strip()
            fasta_dict[name] = seq
        sys.stderr.write(f'Number of FASTA records: {len(fasta_dict)}\n')

        uniprot_mapping_fn = os.path.join(
            out_dir, 'out1.6_uniprot_uniprot_mapping.tab')
        with open(uniprot_mapping_fn, "r") as f:
            for line in f:
                if line.startswith("From"):
                    continue
                fro, to = line.strip().split('\t')
                if fro not in fasta_dict and to in fasta_dict:
                    fasta_dict[fro] = fasta_dict[to]
        sys.stderr.write(f'Number of FASTA records (corrected): {len(fasta_dict)}\n')
        return fasta_dict

    def get_pdbid_to_uniprotid(self, fasta_dict):
        pdbid_set, _, pdbbind_mapping_dict = self.read_INDEX_general_PL_name(
            self.config)
        sys.stderr.write(f'Number of PDB ID to Uniprot ID pairs in PDBBind: {len(pdbbind_mapping_dict)}\n')

        uniprot_mapping_dict = self.pdb_to_uniprot
        sys.stderr.write(f'Number of PDBID to Uniprot ID pairs from UniProt: {len(uniprot_mapping_dict)}\n',)
        uniprot_mapping_dict['4z0e'] = ['A0A024UZE1']
        uniprot_mapping_dict['5ku9'] = ['Q07820']
        uniprot_mapping_dict['5ufs'] = ['Q15466']
        uniprot_mapping_dict['5u51'] = ['Q5NFG1']
        uniprot_mapping_dict['4z0d'] = ['A0A024UZE1']
        uniprot_mapping_dict['4z0f'] = ['A0A024UZE1']

        pdbid_to_uniprotid = {}
        count = 0
        for pdbid in pdbid_set:
            if pdbid in uniprot_mapping_dict and len(uniprot_mapping_dict[pdbid]) == 1:
                pdbid_to_uniprotid[pdbid] = uniprot_mapping_dict[pdbid][0]
            else:
                pdbid_to_uniprotid[pdbid] = pdbbind_mapping_dict[pdbid]
                if pdbbind_mapping_dict[pdbid] not in fasta_dict:
                    count += 1
        sys.stderr.write(f'Merged PDB ID to UniProt ID Pair: {len(pdbid_to_uniprotid)}\n')
        sys.stderr.write(f'PDB ID with no sequence: {count}\n')

        return pdbid_to_uniprotid

    def get_pdbid_to_affinity(self, config):
        pdbid_to_measure, pdbid_to_value = {}, {}   # value: -log [M]
        index_dir = config["PDBBIND"]["INDEX"]
        index_fn = os.path.join(index_dir, "INDEX_general_PL.2018")
        with open(index_fn) as f:
            count_error = 0
            for line in f.readlines():
                if line[0] != '#':
                    lines = line.split('/')[0].strip().split('  ')
                    pdbid = lines[0]
                    if '<' in lines[3] or '>' in lines[3] or '~' in lines[3]:
                        # print lines[3]
                        count_error += 1
                        continue
                    measure = lines[3].split('=')[0]
                    value = float(lines[3].split('=')[1][:-2])
                    unit = lines[3].split('=')[1][-2:]
                    if unit == 'nM':
                        pvalue = -np.log10(value) + 9
                    elif unit == 'uM':
                        pvalue = -np.log10(value) + 6
                    elif unit == 'mM':
                        pvalue = -np.log10(value) + 3
                    elif unit == 'pM':
                        pvalue = -np.log10(value) + 12
                    elif unit == 'fM':
                        pvalue = -np.log10(value) + 15
                    else:
                        print(unit)
                    pdbid_to_measure[pdbid] = measure
                    pdbid_to_value[pdbid] = pvalue
        sys.stderr.write(f'Number of affinity measurement errors (not exact measurement): {count_error}\n')
        return pdbid_to_measure, pdbid_to_value

    def write_all_datafile(self, config, pdbid_to_uniprotid, pdbid_to_ligand, pdbid_to_measure, pdbid_to_value, mol_dict, uniprot_dict):
        out_dir = config["DATA"]["WD"]
        out_fn = os.path.join(out_dir, "out2_pdbbind_all_datafile.tsv")
        count_success = 0
        with open(out_fn, 'w') as fw:
            error_step1, error_step2, error_step3, error_step4 = 0, 0, 0, 0
            for pdbid in pdbid_to_uniprotid:
                if pdbid not in pdbid_to_ligand:
                    error_step1 += 1
                    continue
                if pdbid not in pdbid_to_measure:
                    error_step2 += 1
                    continue
                ligand = pdbid_to_ligand[pdbid]
                if ligand not in mol_dict:
                    sys.stderr.write(f'Missing ligand: {ligand}\n')
                    error_step3 += 1
                    continue
                inchi = Chem.MolToInchi(mol_dict[ligand])

                uniprotid = pdbid_to_uniprotid[pdbid]
                if uniprotid in uniprot_dict:
                    seq = uniprot_dict[uniprotid]
                else:
                    sys.stderr.write(f'Missing UniProt ID: {uniprotid}\n')
                    error_step4 += 1
                    continue

                measure = pdbid_to_measure[pdbid]
                value = pdbid_to_value[pdbid]

                fw.write(pdbid + '\t' + uniprotid + '\t' + ligand + '\t' +
                         inchi + '\t' + seq + '\t' + measure + '\t' + str(value) + '\n')
                count_success += 1
        sys.stderr.write(f'Number of entries in final data file: {count_success}\n')


if __name__ == "__main__":
    data_downloader = DataDownloader(config_fn)
