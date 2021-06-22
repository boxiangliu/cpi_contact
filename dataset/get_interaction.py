from rdkit import Chem
from Bio.PDB import PDBParser, Selection
from Bio.PDB.Polypeptide import three_to_one
import os
import pickle
from utils import DataUtils

class InteractionParser(DataUtils):
    def __init__(self, config_fn, debug=False, download=True, process=True):
        super(InteractionParser, self).__init__(config_fn)
        self.pdbid_list = self.get_pdbid_list(self.config)
        self.pdbid_to_ligand = self.get_pdb_to_ligand(self.config)

    def get_atoms_from_pdb(self, ligand, pdbid):
        '''
        from pdb protein structure, get ligand index list for bond extraction
        '''
        pdb_dir = self.config["DATA"]["PDB"]
        pdb_fn = os.path.join(pdb_dir, pdbid+".pdb")
        p = PDBParser()
        atom_idx_list = []
        atom_name_list = []
        import ipdb; ipdb.set_trace()
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