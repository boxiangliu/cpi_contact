import sys
import pickle
import os
from utils import DataUtils, config_fn
from tqdm import tqdm
import glob
import time

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

    def __init__(self, config_fn, debug=False):
        super(FastaPreparer, self).__init__(config_fn)
        self.interaction_dict = self.read_interaction_dict()
        self.prepare_fasta()
        self.hhblits_commands = self.get_hhblits_commands()
        self.command_list = self.write_hhblits_commands()
        self.run_hhblits_commands()

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
            fasta_fn = os.path.join(out_dir, seq_id + ".fasta")
            if os.path.exists(fasta_fn): 
                continue
            with open(fasta_fn, "w") as f:
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

    def write_hhblits_commands(self):
        hhblits_commands = self.hhblits_commands
        chunks = 800
        per_file = len(hhblits_commands) // chunks + 1
        out_dir = self.config["DATA"]["MSA"]

        i = 0
        file_content = []
        command_list = []
        for command in hhblits_commands:
            if len(file_content) < per_file:
                file_content.append(command)
            else:
                fn = os.path.join(out_dir, f"hhblits_commands_{i:03}")
                command_list.append(fn)
                with open(fn, "w") as f:
                    f.write("\n".join(file_content) + "\n")
                i += 1
                file_content = [command]

        i += 1
        fn = os.path.join(out_dir, f"hhblits_commands_{i:03}")
        command_list.append(fn)
        with open(fn, "w") as f:
            f.write("\n".join(file_content) + "\n")
 
        sys.stderr.write(f"Number of hhblits commands: {i}\n")

        return command_list

    def run_hhblits_commands(self):
        queues = self.config["SLURM"]["QUEUE"]
        if self.debug:
            self.command_list = self.command_list[:2]

        for f in self.command_list:
            time.sleep(1)
            base = os.path.basename(f)
            command = f"sbatch -p {queues} --job-name {base} --wrap 'bash {f}'"
            sys.stderr.write(command + "\n")
            subprocess.run(command, shell=True)

fasta_preparer = FastaPreparer(config_fn, debug=True)
