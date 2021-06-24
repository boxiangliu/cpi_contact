from utils import DataUtils
import pickle
import subprocess
import os
import sys


class Aligner(DataUtils):

    def __init__(self, config_fn, debug=False):
        super(Aligner, self).__init__(config_fn)
        self.pdbid_to_uniprot = self.get_pdbid_to_uniprot()
        self.query_fn, self.target_fn = self.write_seq(self.pdbid_to_uniprot)
        self.align(self.query_fn, self.target_fn)

    def get_pdbid_to_uniprot(self):
        work_dir = self.config["DATA"]["WD"]
        data_fn = os.path.join(work_dir, 'out2_pdbbind_all_datafile.tsv')
        pdbid_to_uniprot = {}
        with open(data_fn) as f:
            for line in f.readlines():
                try:
                    pdbid, uniprotid, ligand, inchi, seq, measure, value = line.strip().split('\t')
                except:
                    sys.stderr.write("[ERROR] line.strip().split('\t')\n")
                    assert 0
                pdbid_to_uniprot[pdbid] = (uniprotid, seq)
        sys.stderr.write(f'Length of pdbid_to_uniprot: {len(pdbid_to_uniprot)}\n')
        return pdbid_to_uniprot

    def write_seq(self, pdbid_to_uniprot):
        '''For PDB'''
        work_dir = self.config["DATA"]["WD"]
        interact_dict_fn = os.path.join(work_dir, 'out4_interaction_dict')
        with open(interact_dict_fn, 'rb') as f:
            interaction_dict = pickle.load(f)

        query_fn = os.path.join(work_dir, 'out6.1_query_pdb.fasta')
        target_fn = os.path.join(work_dir, 'out6.1_target_uniprot_pdb.fasta')
        fw1 = open(query_fn, 'w')
        fw2 = open(target_fn, 'w')
        for name in interaction_dict:
            pdbid = name.split('_')[0]
            if pdbid not in pdbid_to_uniprot:
                continue
            chain_dict = interaction_dict[name]['sequence']
            uniprotid, uniprotseq = pdbid_to_uniprot[pdbid]
            #print('chain_dict', len(chain_dict))
            for chain_id in chain_dict:
                if len(chain_dict[chain_id][0]) == 0:
                    continue
                fw1.write('>' + pdbid + '_' + chain_id + '\n')
                fw1.write(chain_dict[chain_id][0] + '\n')
                fw2.write('>' + pdbid + '_' + chain_id +
                          '_' + uniprotid + '\n')
                fw2.write(uniprotseq + '\n')
        fw1.close()
        fw2.close()
        return query_fn, target_fn

    def align(self, query_fn, target_fn):
        sw_dir = self.config["SRC"]["SW"]
        work_dir = self.config["DATA"]["WD"]
        pyssw_pairwise = os.path.join(sw_dir, "pyssw_pairwise.py")
        out_fn = os.path.join(work_dir, "out6.3_pdb_align.txt")
        command = f"python2 {pyssw_pairwise} -l {sw_dir} -c -p {query_fn} {target_fn} > {out_fn}"
        sys.stderr.write(f"[COMMAND]: {command}\n")
        subprocess.run(command, shell=True)


aligner = Aligner(config_fn)