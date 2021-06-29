import sys
import pickle
import os
from utils import DataUtils, config_fn
from tqdm import tqdm
import glob

# in_fn = sys.argv[1]
# out_dir = sys.argv[2]
# if not os.path.exists(out_dir):
#     os.makedirs(out_dir)

# with open(in_fn, "rb") as f:
#     interaction_dict = pkl.load(f)

# n = 0
# for k, v in interaction_dict.items():
#     n += 1
#     uniprot_id = v["uniprot_id"]
#     pdb_id = k
#     uniprot_seq = v["uniprot_seq"]
#     seq_id = f"{pdb_id}_{uniprot_id}"
#     with open(os.path.join(out_dir, f"{seq_id}.fasta"), "w") as f:
#         f.write(f">{seq_id}\n")
#         f.write(uniprot_seq)

# sys.stderr.write(f"{n} sequences processed.\n")


class FastaPreparer(DataUtils):

    def __init__(self, config_fn):
        super(FastaPreparer, self).__init__(config_fn)
        self.interaction_dict = self.read_interaction_dict()
        self.prepare_fasta()
        self.hhblits_commands = get_hhblits_commands()

    def read_interaction_dict(self):
        work_dir = self.config["DATA"]["WD"]
        in_fn = os.path.join(work_dir, "out7_final_pairwise_interaction_dict")
        with open(in_fn, "rb") as f:
            interaction_dict = pickle.load(f)
        return interaction_dict

    def prepare_fasta(self):
        out_dir = self.config["DATA"]["MSA"]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        n_files = 0
        for k, v in tqdm(self.interaction_dict.items()):
            n_files += 1
            pdb_id = k
            uniprot_id = v["uniprot_id"]
            uniprot_seq = v["uniprot_seq"]
            seq_id = f"{pdb_id}_{uniprot_id}"
            with open(os.path.join(out_dir, seq_id + ".fasta"), "w") as f:
                f.write(f">{seq_id}\n")
                f.write(uniprot_seq)

        sys.stderr.write(f"Number of FASTA files: {n_files}\n")

    def get_hhblits_commands(self):
        out_dir = self.config["DATA"]["MSA"]
        hhblits_database = self.config["HHBLITS"]["DATABASE"]
        commands = []
        for fasta_fn in glob.glob(os.path.join(out_dir, "*.fasta")):
            base = os.path.basename(fasta_fn).replace(".fasta", "")
            command = f"hhblits -i {fasta_fn} -o {out_dir}/{base}.hhr -oa3m {out_dir}/{base}.a3m -d {hhblits_database}"
            commands.append(command)
        return commands
fasta_preparer = FastaPreparer(config_fn)
