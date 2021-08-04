from rdkit import Chem
from Bio.PDB import PDBParser, Selection
from Bio.PDB.Polypeptide import three_to_one
import os
import pickle
from utils import DataUtils, config_fn
import sys
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)


class InteractionParser(DataUtils):

    def __init__(self, config_fn, debug=False):
        super(InteractionParser, self).__init__(config_fn)
        self.pdbid_list = self.get_pdbid_list(self.config)
        self.pdbid_to_ligand = self.get_pdb_to_ligand(self.config)

        if debug:
            self.pdbid_list = self.pdbid_list[:5]

        self.parse(self.pdbid_list, self.pdbid_to_ligand)

    def get_atoms_from_pdb(self, ligand, pdbid):
        '''
        INPUT: 
            pdb ID
            ligand name
        OUTPUT:
            ligand index list
        DESCRIPTION:
            get index for ligand atoms from pdb structure
        '''
        pdb_dir = self.config["DATA"]["PDB"]
        pdb_fn = os.path.join(pdb_dir, pdbid + ".pdb")
        p = PDBParser()
        atom_idx_list = []
        atom_name_list = []
        structure = p.get_structure(pdbid, pdb_fn)
        seq_dict = {}
        for model in structure:
            for chain in model:
                chain_id = chain.get_id()
                id_list = []
                for res in chain:
                    if ligand == res.get_resname():
                        if res.get_id()[0] == ' ':
                            continue
                        for atom in res:
                            atom_idx_list.append(atom.get_serial_number())
                            atom_name_list.append(atom.get_id())
        if len(atom_idx_list) != 0:
            return atom_idx_list, atom_name_list
        else:
            return None, None

    def parse_hydrogen_bonds(self, lines, atom_idx_list, pdbid, ligand):
        atom_idx1, atom_idx2 = int(lines[12]), int(lines[14])
        if atom_idx1 in atom_idx_list and atom_idx2 in atom_idx_list:   # discard ligand-ligand interaction
            return "continue"
        if atom_idx1 in atom_idx_list:
            atom_idx_ligand, atom_idx_protein = atom_idx1, atom_idx2
        elif atom_idx2 in atom_idx_list:
            atom_idx_ligand, atom_idx_protein = atom_idx2, atom_idx1
        else:
            sys.stderr.write(f'[ERROR] Hydrogen bond ({atom_idx1}, {atom_idx2}) in plip result not in atom_idx_list for (PDB ID: {pdbid}, ligand: {ligand})\n')
            return None
        return atom_idx_ligand, atom_idx_protein

    def parse_hydrophobic_interaction(self, lines, atom_idx_list, pdbid, ligand):
        atom_idx_ligand, atom_idx_protein = int(lines[8]), int(lines[9])
        if atom_idx_ligand not in atom_idx_list:
            print(f'[ERROR] Hydrophobic Interactions ({atom_idx_ligand}, {atom_idx_protein}) in plip result not in atom_idx_list for (PDB ID: {pdbid}, ligand: {ligand})\n')
            return None
        return atom_idx_ligand, atom_idx_protein

    def get_pi_interaction(self, lines, atom_idx_list, pdbid, ligand):
        atom_idx_ligand_list = list(
            map(int, lines[12].split(',')))
        if len(set(atom_idx_ligand_list).intersection(set(atom_idx_list))) != len(atom_idx_ligand_list):
            sys.stderr.write(f'[ERROR]: Pi-interaction in plip result not in atom_idx_list for (PDB ID: {pdbid}, ligand: {ligand})\n')
            print(atom_idx_ligand_list)
            return None
        return atom_idx_ligand_list

    def parse_salt_bridges(self, lines, atom_idx_list, pdbid, ligand):
        atom_idx_ligand_list = list(
            set(map(int, lines[11].split(','))))
        if len(set(atom_idx_ligand_list).intersection(set(atom_idx_list))) != len(atom_idx_ligand_list):
            sys.stderr.write(f'[ERROR]: Salt bridge in plip result not in atom_idx_list for (PDB ID: {pdbid}, ligand: {ligand})\n')
            return None
        return atom_idx_ligand_list

    def parse_halogen_bonds(self, lines, atom_idx_list, pdbid, ligand):
        atom_idx1, atom_idx2 = int(lines[11]), int(lines[13])
        if atom_idx1 in atom_idx_list and atom_idx2 in atom_idx_list:   # discard ligand-ligand interaction
            return "continue"
        if atom_idx1 in atom_idx_list:
            atom_idx_ligand, atom_idx_protein = atom_idx1, atom_idx2
        elif atom_idx2 in atom_idx_list:
            atom_idx_ligand, atom_idx_protein = atom_idx2, atom_idx1
        else:
            sys.stderr.write(f'[ERROR]: Halogen bond ({atom_idx1}, {atom_idx2}) in plip result not in atom_idx_list for (PDB ID: {pdbid}, ligand: {ligand})\n')
            return None
        return atom_idx_ligand, atom_idx_protein

    def get_bonds(self, pdbid, ligand, atom_idx_list):
        '''
        DESCRIPTION: 
        Get index of bonds from plip results
        '''
        bond_list = []
        plip_dir = self.config["DATA"]["PLIP"]
        plip_fn = os.path.join(plip_dir, pdbid + '_output.txt')
        with open(plip_fn) as f:
            isheader = False
            for line in f.readlines():

                if line[0] == '*':
                    bond_type = line.strip().replace('*', '')
                    isheader = True
                if line[0] == '|':
                    if isheader:
                        header = line.replace(' ', '').split('|')
                        isheader = False
                        continue
                    lines = line.replace(' ', '').split('|')
                    if ligand not in lines[5]:
                        continue

                    if bond_type != "Salt Bridges":  # Boxiang
                        aa_id, aa_name, aa_chain, ligand_id, ligand_name, ligand_chain = int(
                            lines[1]), lines[2], lines[3], int(lines[4]), lines[5], lines[6]
                    else:  # Boxiang
                        aa_id, aa_name, aa_chain, ligand_id, ligand_name, ligand_chain = int(
                            lines[1]), lines[2], lines[3], int(lines[5]), lines[6], lines[7]  # Boxiang

                    if bond_type in ['Hydrogen Bonds', 'Water Bridges']:
                        hydrogen_bonds = self.parse_hydrogen_bonds(lines, atom_idx_list, pdbid, ligand)
                        if hydrogen_bonds is None:
                            return None
                        elif hydrogen_bonds == "continue":
                            continue
                        atom_idx_ligand, atom_idx_protein = hydrogen_bonds
                        bond_list.append((bond_type + '_' + str(len(bond_list)), aa_chain, aa_name, aa_id,[atom_idx_protein], ligand_chain, ligand_name, ligand_id, [atom_idx_ligand]))

                    elif bond_type == 'Hydrophobic Interactions':
                        hydrophobic_interaction = self.parse_hydrophobic_interaction(lines, atom_idx_list, pdbid, ligand)
                        if hydrophobic_interaction is None:
                            return None
                        atom_idx_ligand, atom_idx_protein = hydrophobic_interaction
                        bond_list.append((bond_type + '_' + str(len(bond_list)), aa_chain, aa_name, aa_id, [atom_idx_protein], ligand_chain, ligand_name, ligand_id, [atom_idx_ligand]))

                    elif bond_type in ['pi-Stacking', 'pi-Cation Interactions']:
                        atom_idx_ligand_list = self.get_pi_interaction(lines, atom_idx_list, pdbid, ligand)
                        if atom_idx_ligand_list is None:
                            return None
                        bond_list.append((bond_type + '_' + str(len(bond_list)), aa_chain, aa_name, aa_id,[], ligand_chain, ligand_name, ligand_id, atom_idx_ligand_list))

                    elif bond_type == 'Salt Bridges':
                        atom_idx_ligand_list = self.parse_salt_bridges(lines, atom_idx_list, pdbid, ligand)
                        if atom_idx_ligand_list is None:
                            return None
                        bond_list.append((bond_type + '_' + str(len(bond_list)), aa_chain, aa_name, aa_id, [], ligand_chain, ligand_name, ligand_id, atom_idx_ligand_list))

                    elif bond_type == 'Halogen Bonds':
                        halogen_bonds = self.parse_halogen_bonds(lines, atom_idx_list, pdbid, ligand)
                        if halogen_bonds is None:
                            return None
                        elif halogen_bonds == "continue":
                            continue
                        atom_idx_ligand, atom_idx_protein = halogen_bonds
                        bond_list.append((bond_type + '_' + str(len(bond_list)), aa_chain, aa_name, aa_id, [atom_idx_protein], ligand_chain, ligand_name, ligand_id, [atom_idx_ligand]))

                    else:
                        print('bond_type', bond_type)
                        print(header)
                        print(lines)
                        return None

        if len(bond_list) != 0:
            return bond_list

    def get_interact_atom_name(self, atom_idx_list, atom_name_list, bond_list):
        interact_atom_name_list = []
        interact_bond_type_list = []
        interact_atom_name_set = set()
        assert len(atom_idx_list) == len(atom_name_list)
        for bond in bond_list:
            for atom_idx in bond[-1]:
                atom_name = atom_name_list[atom_idx_list.index(atom_idx)]
                #if atom_name not in interact_atom_name_set:
                interact_atom_name_set.add(atom_name)
                interact_atom_name_list.append(atom_name)
                interact_bond_type_list.append((atom_name, bond[0]))
        return interact_atom_name_list, interact_bond_type_list

    def get_interact_atom_list(self, name_order_list, atom_name_to_idx_dict, atom_name_to_element_dict, interact_atom_name_list):
        atom_idx_list = []
        atom_name_list = []
        atom_element_list = []
        atom_interact_list = []
        for name in name_order_list:
            idx = atom_name_to_idx_dict[name]
            atom_idx_list.append(idx)
            atom_name_list.append(name)
            atom_element_list.append(atom_name_to_element_dict[name])
            atom_interact_list.append(int(name in interact_atom_name_list))
        return atom_idx_list, atom_name_list, atom_element_list, atom_interact_list

    def get_mol_from_ligandpdb(self, ligand):
        pdb_dir = self.config["DATA"]["PDB"]
        ligand_fn = os.path.join(pdb_dir, ligand + "_ideal.pdb")
        if not os.path.exists(ligand_fn):
            return None, None, None
        name_order_list = []
        name_to_idx_dict, name_to_element_dict = {}, {}
        p = PDBParser()
        try: # Boxiang
            structure = p.get_structure(ligand, ligand_fn)
            for model in structure:
                for chain in model:
                    chain_id = chain.get_id()
                    for res in chain:
                        if ligand == res.get_resname():
                            # print(ligand, res.get_resname(), res.get_full_id())
                            for atom in res:
                                name_order_list.append(atom.get_id())
                                name_to_element_dict[atom.get_id()] = atom.element
                                name_to_idx_dict[atom.get_id()] = atom.get_serial_number()-1
            # print('check', name_to_idx_dict.items())
            if len(name_to_idx_dict) == 0:
                return None, None, None
            return name_order_list, name_to_idx_dict, name_to_element_dict
        except: # Boxiang 
            return None, None, None # Boxiang 

    def get_interact_atom_name(self, atom_idx_list, atom_name_list,bond_list):
        interact_atom_name_list = []
        interact_bond_type_list = []
        interact_atom_name_set = set()
        assert len(atom_idx_list) == len(atom_name_list)
        for bond in bond_list:
            for atom_idx in bond[-1]:
                atom_name = atom_name_list[atom_idx_list.index(atom_idx)]
                #if atom_name not in interact_atom_name_set:
                interact_atom_name_set.add(atom_name)
                interact_atom_name_list.append(atom_name)
                interact_bond_type_list.append((atom_name, bond[0]))
        return interact_atom_name_list, interact_bond_type_list

    def get_seq(self, pdbid):
        p = PDBParser()
        pdb_dir = self.config["DATA"]["PDB"]
        pdb_fn = os.path.join(pdb_dir, pdbid+'.pdb')
        structure = p.get_structure(pdbid, pdb_fn)
        seq_dict = {}
        idx_to_aa_dict = {}
        for model in structure:
            for chain in model:
                chain_id = chain.get_id()
                if chain_id == ' ':
                    continue
                seq = ''
                id_list = []
                for res in chain:
                    if res.get_id()[0] != ' ' or res.get_id()[2] != ' ':   # remove HETATM
                        continue
                    try:
                        seq+=three_to_one(res.get_resname())
                        idx_to_aa_dict[chain_id+str(res.get_id()[1])+res.get_id()[2].strip()] = three_to_one(res.get_resname())
                    except:
                        print('unexpected aa name', res.get_resname())
                    id_list.append(res.get_id()[1])
                seq_dict[chain_id] = (seq,id_list)
        return seq_dict, idx_to_aa_dict

    def get_interact_residue(self, idx_to_aa_dict, bond_list):
        interact_residue = []
        for bond in bond_list:
            if bond[1]+str(bond[3]) not in idx_to_aa_dict:
                continue
            aa = idx_to_aa_dict[bond[1]+str(bond[3])]
            assert three_to_one(bond[2]) == aa
            interact_residue.append((bond[1]+str(bond[3]), aa, bond[0]))
        if len(interact_residue) != 0:
            return interact_residue
        else:
            return None

    def parse(self, pdbid_list, pdbid_to_ligand):
        no_valid_ligand = 0
        no_such_ligand_in_pdb_error = 0
        no_interaction_detected_error = 0
        no_ideal_pdb_error = 0
        empty_atom_interact_list = 0
        protein_seq_error = 0

        i = 0
        interaction_dict = {}
        for pdbid in pdbid_list:
            i += 1
            sys.stderr.write(f"{i}\t{pdbid}\n")
            if pdbid not in pdbid_to_ligand:
                no_valid_ligand += 1
                continue
            ligand = pdbid_to_ligand[pdbid]

            # get bond
            atom_idx_list, atom_name_list = self.get_atoms_from_pdb(
                ligand, pdbid)  # for bond atom identification
            if atom_idx_list is None:
                no_such_ligand_in_pdb_error += 1
                sys.stderr.write(f'Ligand {ligand} not in pdb structure {pdbid}\n')
                continue
            bond_list = self.get_bonds(pdbid, ligand, atom_idx_list)
            if bond_list is None:
                print('empty bond list: pdbid', pdbid, 'ligand',
                      ligand, 'atom_idx_list', len(atom_idx_list))
                no_interaction_detected_error += 1
                continue
            interact_atom_name_list, interact_bond_type_list = self.get_interact_atom_name(
                atom_idx_list, atom_name_list, bond_list)

            name_order_list, atom_name_to_idx_dict, atom_name_to_element_dict = self.get_mol_from_ligandpdb(
                ligand)
            if atom_name_to_idx_dict == None:
                no_ideal_pdb_error += 1
                continue
            atom_idx_list, atom_name_list, atom_element_list, atom_interact_list \
                = self.get_interact_atom_list(name_order_list, atom_name_to_idx_dict, atom_name_to_element_dict, interact_atom_name_list)
            if len(atom_idx_list) == 0:
                empty_atom_interact_list += 1
                continue

            # get sequence interaction information
            seq_dict, idx_to_aa_dict = self.get_seq(pdbid)
            interact_residue_list = self.get_interact_residue(
                idx_to_aa_dict, bond_list)
            if interact_residue_list is None:
                protein_seq_error += 1
                continue

            interaction_dict[pdbid + '_' + ligand] = {}
            interaction_dict[pdbid + '_' + ligand]['bond'] = bond_list
            interaction_dict[pdbid + '_' + ligand]['atom_idx'] = atom_idx_list
            interaction_dict[pdbid + '_' +
                             ligand]['atom_name'] = atom_name_list
            interaction_dict[pdbid + '_' +
                             ligand]['atom_element'] = atom_element_list
            interaction_dict[pdbid + '_' +
                             ligand]['atom_interact'] = atom_interact_list
            interaction_dict[pdbid + '_' +
                             ligand]['atom_bond_type'] = interact_bond_type_list

            interaction_dict[pdbid + '_' + ligand]['sequence'] = seq_dict
            interaction_dict[pdbid + '_' +
                             ligand]['residue_interact'] = interact_residue_list

        sys.stderr.write(f'Length of interaction_dict: {len(interaction_dict)}\n')
        sys.stderr.write(f'No ligand found: {no_valid_ligand}\n')
        sys.stderr.write(f'Ligand not in PDB structure: {no_such_ligand_in_pdb_error}\n')
        sys.stderr.write(f'No interaction between ligand and protein: {no_interaction_detected_error}\n')
        sys.stderr.write(f'Ligand not found in Components-pub.sdf: {no_ideal_pdb_error}\n')
        sys.stderr.write(f'empty_atom_interact_list: {empty_atom_interact_list}\n')
        sys.stderr.write(f'Interaction residue not found: {protein_seq_error}\n')

        out_dir = self.config["DATA"]["WD"]
        with open(os.path.join(out_dir, 'out4_interaction_dict'), 'wb') as f:
            pickle.dump(interaction_dict, f, protocol=0)

interaction_parser = InteractionParser(config_fn)
