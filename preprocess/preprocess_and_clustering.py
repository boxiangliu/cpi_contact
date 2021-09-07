import sys
from utils import DataUtils, config_fn, aa_list, elem_list
import os
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from collections import defaultdict
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage, single
from scipy.spatial.distance import pdist
import torch
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Process and clustering")
parser.add_argument("measure",
                    type=str,
                    help="one of IC50 or KIKD")
parser.add_argument("msa_mode",
                    type=str,
                    help="one of ESM or AF2")


def pickle_dump(dictionary, file_name):
    pickle.dump(dict(dictionary), open(file_name, 'wb'), protocol=0)


class Preprocessor(DataUtils):
    def __init__(self, config_fn, measure="IC50", msa_mode="ESM", debug=False):
        super(Preprocessor, self).__init__(config_fn)
        self.msa_mode = msa_mode
        self.debug = debug
        self.max_nb = 6
        self.word_dict = self.get_word_dict()
        self.bond_dict = defaultdict(lambda: len(self.bond_dict))
        self.atom_dict = defaultdict(lambda: len(self.atom_dict))

        self.set_measure(measure)
        self.mol_dict = self.get_mol_dict()
        self.interaction_dict = self.get_interaction_dict()
        self.pair_info_dict = self.get_pair_info_dict()
        self.get_input()
        self.save_data()
        self.clustering()

    def set_measure(self, measure):
        assert measure in ["IC50", "KIKD"]
        self.MEASURE = measure
        sys.stderr.write(f"Creating dataset for measurement: {measure}\n")

    def get_mol_dict(self):
        mol_dict_fn = self.config["PDB"]["MOL_DICT"]
        coordinate_fn = self.config["PDB"]["COORDINATE"]

        if os.path.exists(mol_dict_fn):
            with open(mol_dict_fn, "rb") as f:
                mol_dict = pickle.load(f)
        else:
            mol_dict = {}
            mols = Chem.SDMolSupplier(coordinate_fn)
            for m in mols:
                if m is None:
                    continue
                name = m.GetProp("_Name")
                mol_dict[name] = m
        with open(mol_dict_fn, 'wb') as f:
            pickle.dump(mol_dict, f)

        sys.stderr.write("Finished loading mol_dict.\n")
        return mol_dict

    def get_interaction_dict(self):
        wd = self.config["DATA"]["WD"]
        interaction_dict_fn = os.path.join(wd, "out7_final_pairwise_interaction_dict")
        with open(interaction_dict_fn, "rb") as f:
            interaction_dict = pickle.load(f)

        sys.stderr.write("Finished loading interaction dict.\n")
        return interaction_dict

    def get_word_dict(self):
        word_dict = defaultdict(lambda: len(word_dict))
        for aa in aa_list:
            word_dict[aa]
        word_dict["X"]
        sys.stderr.write("Finished getting word dict.\n")
        return word_dict

    def get_pairwise_label(self, pdbid, interaction_dict, mol):

        if pdbid in interaction_dict:
            sdf_element = np.array([atom.GetSymbol().upper() for atom in mol.GetAtoms()])
            atom_element = np.array(interaction_dict[pdbid]['atom_element'], dtype=str)
            atom_name_list = np.array(interaction_dict[pdbid]['atom_name'], dtype=str)
            atom_interact = np.array(interaction_dict[pdbid]['atom_interact'], dtype=int)
            nonH_position = np.where(atom_element != ('H'))[0]
            try:
                assert sum(atom_element[nonH_position] != sdf_element) == 0
            except:
                return False, np.zeros((1,1))

            atom_name_list = atom_name_list[nonH_position].tolist()
            pairwise_mat = np.zeros((len(nonH_position), len(interaction_dict[pdbid]['uniprot_seq'])), dtype=np.int32)
            for atom_name, bond_type in interaction_dict[pdbid]['atom_bond_type']:
                atom_idx = atom_name_list.index(str(atom_name))
                assert atom_idx < len(nonH_position)

                for seq_idx, bond_type_seq in interaction_dict[pdbid]['residue_bond_type']:
                    if bond_type == bond_type_seq:
                        pairwise_mat[atom_idx, seq_idx] = 1

            if len(np.where(pairwise_mat != 0)[0]) != 0:
                 # return pairwise_mask and pairwise_mat
                return True, pairwise_mat

        return False, np.zeros((1,1))

    def get_pair_info_dict(self):
        MEASURE = self.MEASURE
        i = 0
        pair_info_dict = {}
        wd = self.config["DATA"]["WD"]
        pdbbind_all_datafile = os.path.join(wd, "out2_pdbbind_all_datafile.tsv")

        f = open(pdbbind_all_datafile)
        sys.stderr.write('Step 2/5, generating labels...\n')
        for line in f.readlines():
            i += 1
            if i % 1000 == 0:
                sys.stderr.write(f'processed sample num: {i}\n')
            pdbid, pid, cid, inchi, seq, measure, label = line.strip().split('\t')
            
            # filter interaction type and invalid molecules
            if MEASURE == 'All':
                pass
            elif MEASURE == 'KIKD':
                if measure not in ['Ki', 'Kd']:
                    continue
            elif measure != MEASURE:
                continue

            if cid not in self.mol_dict:
                sys.stderr.write('ligand not in mol_dict\n')
                continue
            mol = self.mol_dict[cid]
            
            # get labels
            value = float(label)
            pairwise_mask, pairwise_mat = self.get_pairwise_label(pdbid, self.interaction_dict, mol)
            
            # handle the condition when multiple PDB entries have the same Uniprot ID and Inchi
            if inchi+' '+pid not in pair_info_dict:
                pair_info_dict[inchi+' '+pid] = [pdbid, cid, pid, value, mol, seq, pairwise_mask, pairwise_mat]
            else:
                if pair_info_dict[inchi+' '+pid][6]:
                    if pairwise_mask and pair_info_dict[inchi+' '+pid][3] < value:
                        pair_info_dict[inchi+' '+pid] = [pdbid, cid, pid, value, mol, seq, pairwise_mask, pairwise_mat]
                else:
                    if pair_info_dict[inchi+' '+pid][3] < value:
                        pair_info_dict[inchi+' '+pid] = [pdbid, cid, pid, value, mol, seq, pairwise_mask, pairwise_mat]
        f.close()
        return pair_info_dict

    def onek_encoding_unk(self, x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))


    def atom_features(self, atom):
        onek_encoding_unk = self.onek_encoding_unk

        return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list) 
                + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
                + onek_encoding_unk(atom.GetExplicitValence(), [1,2,3,4,5,6])
                + onek_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5])
                + [atom.GetIsAromatic()], dtype=np.float32)


    def bond_features(self, bond):
        bt = bond.GetBondType()
        return np.array([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, \
        bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)


    def Mol2Graph(self, mol):
        # convert molecule to GNN input
        max_nb = self.max_nb
        atom_features = self.atom_features
        bond_features = self.bond_features
        idxfunc=lambda x:x.GetIdx()

        n_atoms = mol.GetNumAtoms()
        assert mol.GetNumBonds() >= 0

        n_bonds = max(mol.GetNumBonds(), 1)
        fatoms = np.zeros((n_atoms,), dtype=np.int32) #atom feature ID
        fbonds = np.zeros((n_bonds,), dtype=np.int32) #bond feature ID
        atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
        bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
        num_nbs = np.zeros((n_atoms,), dtype=np.int32)
        num_nbs_mat = np.zeros((n_atoms,max_nb), dtype=np.int32)

        for atom in mol.GetAtoms():
            idx = idxfunc(atom)
            fatoms[idx] = self.atom_dict[''.join(str(x) for x in atom_features(atom).astype(int).tolist())] 

        for bond in mol.GetBonds():
            a1 = idxfunc(bond.GetBeginAtom())
            a2 = idxfunc(bond.GetEndAtom())
            idx = bond.GetIdx()
            fbonds[idx] = self.bond_dict[''.join(str(x) for x in bond_features(bond).astype(int).tolist())] 
            try:
                atom_nb[a1,num_nbs[a1]] = a2
                atom_nb[a2,num_nbs[a2]] = a1
            except:
                return [], [], [], [], []
            bond_nb[a1,num_nbs[a1]] = idx
            bond_nb[a2,num_nbs[a2]] = idx
            num_nbs[a1] += 1
            num_nbs[a2] += 1
            
        for i in range(len(num_nbs)):
            num_nbs_mat[i,:num_nbs[i]] = 1

        return fatoms, fbonds, atom_nb, bond_nb, num_nbs_mat

    def Protein2Sequence(self, sequence, ngram=1):
        # convert sequence to CNN input
        sequence = sequence.upper()
        word_list = [sequence[i:i+ngram] for i in range(len(sequence)-ngram+1)]
        output = []
        for word in word_list:
            if word not in aa_list:
                output.append(self.word_dict['X'])
            else:
                output.append(self.word_dict[word])
        if ngram == 3:
            output = [-1]+output+[-1] # pad
        return np.array(output, np.int32)

    def get_input(self):
        sys.stderr.write('Step 3/5, generating inputs...\n')
        pair_info_dict = self.pair_info_dict
        self.wlnn_train_list = []
        self.valid_value_list = []
        self.valid_cid_list = []
        self.valid_pid_list = []
        self.valid_pairwise_mask_list = []
        self.valid_pairwise_mat_list = []
        self.mol_inputs, self.seq_inputs = [], []
        self.msa_features_dict = {}
        self.msa_features = []
        n_msa_feature_not_found = 0

        pid_list = [pair_info_dict[k][2] for k in pair_info_dict]
        if self.debug:
            pid_list = pid_list[:20]
        if self.msa_mode == "ESM":
            msa_feature_dir = self.config["DATA"]["MSA_FEATURES"]
            for pid in tqdm(set(pid_list)):
                msa_feature_fn = os.path.join(msa_feature_dir, pid + ".pt")
                if not os.path.exists(msa_feature_fn):
                    continue
                msa_feature = torch.load(msa_feature_fn)
                # dimension: B x MSA x SEQ x HIDDEN
                self.msa_features_dict[pid] = msa_feature["representations"][0,0,1:,:].detach().numpy()

        elif self.msa_mode == "AF2":
            msa_features_dir = self.config["DATA"]["AF2"]
            for pid in tqdm(set(pid_list)):
                msa_features_fn = os.path.join(msa_feature_dir, pid, "result_model_1.pkl")
                if not os.path.exists(msa_feature_fn):
                    continue
                with open(msa_feature_fn, "rb") as f:
                    msa_feature = pickle.load(f)
                self.msa_features_dict[pid] = msa_feature["representations"]["msa_first_row"] 

        # get inputs
        for item in tqdm(pair_info_dict):
            pdbid, cid, pid, value, mol, seq, pairwise_mask, pairwise_mat = pair_info_dict[item]
            fa, fb, anb, bnb, nbs_mat = self.Mol2Graph(mol)

            if np.array_equal(fa,[]):
                sys.stderr.write(f'num of neighbor > 6, {cid}\n')
                continue

            if pid not in self.msa_features_dict:
                sys.stderr.write(f"{pid}: MSA not found\n")
                n_msa_feature_not_found += 1
                continue

            self.msa_features.append(self.msa_features_dict[pid])
            self.mol_inputs.append([fa, fb, anb, bnb, nbs_mat])
            self.seq_inputs.append(self.Protein2Sequence(seq,ngram=1))
            self.valid_value_list.append(value)
            self.valid_cid_list.append(cid)
            self.valid_pid_list.append(pid)
            self.valid_pairwise_mask_list.append(pairwise_mask)
            self.valid_pairwise_mat_list.append(pairwise_mat)
            self.wlnn_train_list.append(pdbid)
        sys.stderr.write(f"{n_msa_feature_not_found} MSA features not found\n")

    def save_data(self):
        # get data pack
        wlnn_train_list = self.wlnn_train_list
        valid_value_list = self.valid_value_list
        valid_cid_list = self.valid_cid_list
        valid_pid_list = self.valid_pid_list
        valid_pairwise_mask_list = self.valid_pairwise_mask_list
        valid_pairwise_mat_list = self.valid_pairwise_mat_list
        mol_inputs = self.mol_inputs
        seq_inputs = self.seq_inputs
        msa_features = self.msa_features
        msa_features_dict = self.msa_features_dict
        MEASURE = self.MEASURE
        if self.msa_mode == "ESM":
            preprocessed_dir = self.config["DATA"]["PREPROCESSED"]
        elif self.msa_mode == "AF2":
            preprocessed_dir = self.config["DATA"]["PREPROCESSED_AF2"]

        if not os.path.exists(preprocessed_dir):
            os.makedirs(preprocessed_dir)

        fa_list, fb_list, anb_list, bnb_list, nbs_mat_list = zip(*mol_inputs)
        data_pack = [np.array(fa_list), np.array(fb_list), np.array(anb_list), np.array(bnb_list), \
                     np.array(nbs_mat_list), np.array(seq_inputs), np.array(valid_value_list), \
                     np.array(valid_cid_list), np.array(valid_pid_list), np.array(valid_pairwise_mask_list), \
                     np.array(valid_pairwise_mat_list), np.array(msa_features)]
        
        # save data
        with open(os.path.join(preprocessed_dir, 'pdbbind_all_combined_input_'+MEASURE), 'wb') as f:
            pickle.dump(data_pack, f, protocol=0)
        
        self.data_pack = data_pack
        np.save(os.path.join(preprocessed_dir, 'wlnn_train_list_'+MEASURE), wlnn_train_list)
        
        pickle_dump(self.atom_dict, os.path.join(preprocessed_dir, 'pdbbind_all_atom_dict_'+MEASURE))
        pickle_dump(self.bond_dict, os.path.join(preprocessed_dir, 'pdbbind_all_bond_dict_'+MEASURE))
        pickle_dump(self.word_dict, os.path.join(preprocessed_dir, 'pdbbind_all_word_dict_'+MEASURE))


    def get_fps(self, mol_list):
        fps = []
        for mol in mol_list:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024,useChirality=True)
            fps.append(fp)
        return fps

    def calculate_sims(self, fps1, fps2, simtype='tanimoto'):
        sim_mat = np.zeros((len(fps1),len(fps2)))
        for i in range(len(fps1)):
            fp_i = fps1[i]
            if simtype == 'tanimoto':
                sims = DataStructs.BulkTanimotoSimilarity(fp_i,fps2)
            elif simtype == 'dice':
                sims = DataStructs.BulkDiceSimilarity(fp_i,fps2)
            sim_mat[i,:] = sims
        return sim_mat

    def compound_clustering(self, ligand_list, mol_list):
        sys.stderr.write('start compound clustering...\n')
        fps = self.get_fps(mol_list)
        sim_mat = self.calculate_sims(fps, fps)
        sys.stderr.write(f'compound sim mat: {sim_mat.shape}\n')
        preprocessed_dir = self.config["DATA"]["PREPROCESSED"]

        C_dist = pdist(fps, 'jaccard')
        C_link = single(C_dist)
        for thre in [0.3, 0.4, 0.5, 0.6]:
            C_clusters = fcluster(C_link, thre, 'distance')
            len_list = []
            for i in range(1,max(C_clusters)+1):
                len_list.append(C_clusters.tolist().count(i))
            sys.stderr.write(f'thresold: {thre}; total num of compounds: {len(ligand_list)}; num of clusters: {max(C_clusters)}; max_length: {max(len_list)}\n')
            C_cluster_dict = {ligand_list[i]:C_clusters[i] for i in range(len(ligand_list))}
            with open(os.path.join(preprocessed_dir, self.MEASURE+'_compound_cluster_dict_'+str(thre)),'wb') as f:
                pickle.dump(C_cluster_dict, f, protocol=0)


    def protein_clustering(self, protein_list):
        preprocessed_dir = self.config["DATA"]["PREPROCESSED"]
        alignment_score_fn = os.path.join(self.config["DATA"]["WD"], "alignment_scores.pkl")
        with open(alignment_score_fn, "rb") as f:
            alignment_score = pickle.load(f)
        calc_dist = lambda p_i, p_j: 1 - alignment_score[p_i][p_j] / \
                    (np.sqrt(alignment_score[p_i][p_i]) * np.sqrt(alignment_score[p_j][p_j]))


        # Remove proteins not in alignment_score
        protein_list = [x for x in protein_list if x in alignment_score] 
        n_protein = len(protein_list)
        P_dist = []
        for i in range(n_protein):
            for j in range(i+1, n_protein):
                p_i = protein_list[i]
                p_j = protein_list[j]
                P_dist.append(calc_dist(p_i, p_j))
        P_dist = np.array(P_dist)
        P_link = single(P_dist)
        # Address numerical precision problem
        P_link[P_link < 0] = 0 
        for thre in [0.3, 0.4, 0.5, 0.6]:
            P_clusters = fcluster(P_link, thre, 'distance')
            len_list = []
            for i in range(1,max(P_clusters)+1):
                len_list.append(P_clusters.tolist().count(i))
            sys.stderr.write(f'threshold: {thre}; total num of proteins: {len(protein_list)}; num of clusters: {max(P_clusters)}; max length: {max(len_list)}\n')
            P_cluster_dict = {protein_list[i]:P_clusters[i] for i in range(len(protein_list))}
            with open(os.path.join(preprocessed_dir, self.MEASURE+'_protein_cluster_dict_'+str(thre)),'wb') as f:
                pickle.dump(P_cluster_dict, f, protocol=0)


    def clustering(self):
        sys.stderr.write('Step 5/5, clustering...\n')
        # compound clustering
        compound_list = list(set(self.valid_cid_list))
        mol_list = [self.mol_dict[ligand] for ligand in compound_list]
        self.compound_clustering(compound_list, mol_list)

        # protein clustering
        protein_list = list(set(self.valid_pid_list))
        self.protein_clustering(protein_list)


def main():
    args = parser.parse_args()
    preprocessor = Preprocessor(config_fn, measure=args.measure, msa_mode=args.msa_mode, debug=False)