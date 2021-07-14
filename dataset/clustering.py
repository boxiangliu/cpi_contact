import sys
from utils import DataUtils, config_fn, aa_list, elem_list
import os
import pickle
from rdkit import Chem
from collections import defaultdict
import numpy as np

class Preprocessor(DataUtils):
    def __init__(self, config_fn, measure="IC50"):
        super(Preprocessor, self).__init__(config_fn)
        self.max_nb = 6
        self.word_dict = self.get_word_dict()
        self.bond_dict = defaultdict(lambda: len(self.bond_dict))
        self.atom_dict = defaultdict(lambda: len(self.atom_dict))

        self.set_measure(measure)
        self.mol_dict = self.get_mol_dict()
        self.interaction_dict = self.get_interaction_dict()
        self.pair_info_dict = self.get_pair_info_dict()
        self.get_input()


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
        
        # get inputs
        for item in pair_info_dict:
            pdbid, cid, pid, value, mol, seq, pairwise_mask, pairwise_mat = pair_info_dict[item]
            fa, fb, anb, bnb, nbs_mat = self.Mol2Graph(mol)

            if np.array_equal(fa,[]):
                sys.stderr.write(f'num of neighbor > 6, {cid}\n')
                continue
            self.mol_inputs.append([fa, fb, anb, bnb, nbs_mat])
            self.seq_inputs.append(self.Protein2Sequence(seq,ngram=1))
            self.valid_value_list.append(value)
            self.valid_cid_list.append(cid)
            self.valid_pid_list.append(pid)
            self.valid_pairwise_mask_list.append(pairwise_mask)
            self.valid_pairwise_mat_list.append(pairwise_mat)
            self.wlnn_train_list.append(pdbid)


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
        MEASURE = self.MEASURE

        fa_list, fb_list, anb_list, bnb_list, nbs_mat_list = zip(*mol_inputs)
        data_pack = [np.array(fa_list), np.array(fb_list), np.array(anb_list), np.array(bnb_list), np.array(nbs_mat_list), np.array(seq_inputs), \
        np.array(valid_value_list), np.array(valid_cid_list), np.array(valid_pid_list), np.array(valid_pairwise_mask_list), np.array(valid_pairwise_mat_list)]
        
        # save data
        with open('../preprocessing/pdbbind_all_combined_input_'+MEASURE, 'wb') as f:
            pickle.dump(data_pack, f, protocol=0)
        
        np.save('../preprocessing/wlnn_train_list_'+MEASURE, wlnn_train_list)
        
        pickle_dump(atom_dict, '../preprocessing/pdbbind_all_atom_dict_'+MEASURE)
        pickle_dump(bond_dict, '../preprocessing/pdbbind_all_bond_dict_'+MEASURE)
        pickle_dump(word_dict, '../preprocessing/pdbbind_all_word_dict_'+MEASURE)


preprocessor = Preprocessor(config_fn)
