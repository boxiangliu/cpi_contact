import sys
import pickle
import os
from utils import DataUtils, config_fn
from tqdm import tqdm
import glob
import time
import subprocess
import esm
from Bio import SeqIO
import itertools
from typing import List, Tuple
import string
import multiprocessing as mp
import torch
from collections import Counter
import click

class FastaPreparer(DataUtils):

    def __init__(self, config_fn, debug=False):
        super(FastaPreparer, self).__init__(config_fn)
        self.debug = debug
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


class MSAUtils(DataUtils):

    def __init__(self, config):
        super(MSAUtils, self).__init__(config_fn)

    @property
    def translation(self):
        # This is an efficient way to delete lowercase characters and insertion
        # characters from a string
        deletekeys = dict.fromkeys(string.ascii_lowercase)
        deletekeys["."] = None
        deletekeys["*"] = None
        translation = str.maketrans(deletekeys)
        return translation

    def read_sequence(self, filename: str) -> Tuple[str, str]:
        """ Reads the first (reference) sequences from a fasta or MSA file."""
        record = next(SeqIO.parse(filename, "fasta"))
        return record.description, str(record.seq)

    def remove_insertions(self, sequence: str) -> str:
        """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
        return sequence.translate(self.translation)

    def read_msa(self, filename: str, nseq: int) -> List[Tuple[str, str]]:
        """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
        return [(record.description, self.remove_insertions(str(record.seq)))
                for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]


class MSAFeatureExtractor(MSAUtils):

    def __init__(self, config, model):
        super(MSAFeatureExtractor, self).__init__(config_fn)
        self.model = model
        self.model, self.alphabet, self.converter = self.get_model(model)

    def get_model(self, model, eval=False):
        if model == "esm_msa1_t12_100M_UR50S":
            model, alphabet = esm.pretrained.esm_msa1_t12_100M_UR50S()
        elif model == "esm1b_t33_650M_UR50S":
            model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        else:
            raise ValueError(f"{model} not implemented!")
        if eval:
            model = model.eval()

        return model, alphabet, alphabet.get_batch_converter()

    def extract_from_msa(self, a3m_fn):
        if isinstance(a3m_fn, str):
            msa_data = [self.read_msa(a3m_fn, 64)]
        elif isinstance(a3m_fn, list):
            msa_data = [self.read_msa(fn, 64) for fn in a3m_fn]

        id_ = os.path.basename(a3m_fn).replace(".a3m", "")
        msa_batch_labels, msa_batch_strs, msa_batch_tokens = self.converter(
            msa_data)
        # Remove batch with lengths bigger than 1024:
        if msa_batch_tokens.shape[2] > 1024:
            sys.stderr.write(f"{id_} is longer than 1024.\n")
            return None
        results = self.model.forward(
            msa_batch_tokens, need_head_weights=True, repr_layers=[12])
        return {"id": id_, "representations": results["representations"][12], "row_attentions": results["row_attentions"][:, -1, :, :, :]}

    def extract_and_save(self, a3m_fn):
        out_dir = self.config["DATA"]["MSA_FEATURES"]
        id_ = os.path.basename(a3m_fn).replace(".a3m", "")
        out_fn = os.path.join(out_dir, id_ + ".pt")

        if os.path.exists(out_fn):
            sys.stderr.write(f"{out_fn} already on disk.\n")
            return "already on disk"
        else:
            results = self.extract_from_msa(a3m_fn)
            if results == None:
                return "too long"
            torch.save(results, out_fn)
            sys.stderr.write(f"{out_fn} saved to disk.\n")
            return "saved"

@click.command()
@click.option("--fn_list", default=None, help="File list")
@click.option("--debug", default=False, help="Debug")
def main(fn_list=None, debug=False):
    # fasta_preparer = FastaPreparer(config_fn)
    msa_feature_extractor = MSAFeatureExtractor(
        config_fn, "esm_msa1_t12_100M_UR50S")

    msa_dir = msa_feature_extractor.config["DATA"]["MSA"]

    if fn_list is None:
        fn_list = []
        for fn in glob.glob(os.path.join(msa_dir, "*.a3m")):
            fn_list.append(fn)
    elif isinstance(fn_list, list):
        pass
    elif isinstance(fn_list, str):
        with open(fn_list) as f:
            fn_list = f.readlines()

    if debug:
        fn_list = fn_list[:5]
    sys.stderr.write(f"Files read: {len(fn_list)}\n")

    status = Counter()
    status['n'] = len(fn_list)
    for fn in fn_list:
        res = msa_feature_extractor.extract_and_save(fn)
        status[res] += 1

    sys.stderr.write(f"Total proteins processed: {status['n']}\n")
    sys.stderr.write(f"Protein > 1024 residues: {status['too long']}\n")
    sys.stderr.write(f"Results saved to disk: {status['saved']}\n")
    sys.stderr.write(f"Results already on disk: {status['already on disk']}\n")

    return status

if __name__ == "__main__":
    status = main()
