import sys
from utils import DataUtils, config_fn, aa_list
import os
import pickle
from rdkit import Chem
from collections import defaultdict

class Preprocessor(DataUtils):
    def __init__(self, config_fn, measure="IC50"):
        super(Preprocessor, self).__init__(config_fn)
        self.set_measure(measure)
        self.mol_dict = self.get_mol_dict()
        self.interaction_dict = self.get_interaction_dict()
        self.word_dict = self.get_word_dict()

    def set_measure(self, measure):
        assert measure in ["IC50", "KIKD"]
        self.measure = measure
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
        return word_dict
preprocessor = Preprocessor(config_fn)