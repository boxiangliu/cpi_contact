from utils import DataUtils, config_fn
import os
import sys
from tqdm import tqdm
import subprocess
from glob import glob
from collections import defaultdict

class Aligner(DataUtils):
    def __init__(self, config_fn, debug=False):
        super(Aligner, self).__init__(config_fn)
        self.debug = debug
        self.id_to_seq = self.get_uniprot_id_to_seq()
        self.fasta_fn_list = self.save_seq_to_files()
        self.align_all_pairs()

    def get_uniprot_id_to_seq(self):
        wd = self.config["DATA"]["WD"]
        fasta_fn = os.path.join(wd, "out6.1_target_uniprot_pdb.fasta")
        id_to_seq = dict()
        with open(fasta_fn, "r") as f:
            for line in f:
                if line.startswith(">"):
                    split_line = line.strip().split("_")
                    uniprot_id = split_line[2]
                else:
                    seq = line.strip()
                    if uniprot_id in id_to_seq:
                        assert id_to_seq[uniprot_id] == seq
                    else:
                        id_to_seq[uniprot_id] = seq
        sys.stderr.write(f"Number of UniProt seqs: {len(id_to_seq)}\n")
        return id_to_seq

    def save_seq_to_files(self):
        wd = self.config["DATA"]["WD"]
        out_dir = os.path.join(wd, "one_fasta_per_file")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_fn_list = []
        sys.stderr.write("Writing to disk...\n")
        for id_, seq in tqdm(self.id_to_seq.items()):
            out_fn = os.path.join(out_dir, id_ + ".fasta")
            out_fn_list.append(out_fn)
            if os.path.exists(out_fn):
                continue
            with open(out_fn, "w") as f:
                f.write(f">{id_}\n")
                f.write(f"{seq}\n")

        with open(os.path.join(wd, "out6.1_target_uniprot_pdb_nr.fasta"), "w") as f:
            for id_, seq in tqdm(self.id_to_seq.items()):
                f.write(f">{id_}\n")
                f.write(f"{seq}\n")

        return out_fn_list

    def align_all_pairs(self):
        queue = self.config["SLURM"]["QUEUE"]
        sw_dir = self.config["SRC"]["SW"]
        wd = self.config["DATA"]["WD"]
        all_fasta_fn = os.path.join(wd, "out6.1_target_uniprot_pdb_nr.fasta")

        if self.debug:
            self.fasta_fn_list = self.fasta_fn_list[:5]

        for fasta_fn in self.fasta_fn_list:
            base = os.path.basename(fasta_fn).replace(".fasta", "")
            out_fn = os.path.join(wd, "one_fasta_per_file", base + "_align.txt")
            py_cmd = f"python2 {sw_dir}/pyssw.py -l {sw_dir} -c -p {fasta_fn} {all_fasta_fn} > {out_fn}"
            command = f"sbatch -p {queue} --job-name {base} --wrap '{py_cmd}'"
            sys.stderr.write(command + "\n")
            subprocess.run(command, shell=True)


class AlignmentParser(DataUtils):
    def __init__(self, config_fn, debug=False):
        super(AlignmentParser, self).__init__(config_fn)
        self.score_dict = self.parse()
        self.save()

    def parse(self):
        wd = self.config["DATA"]["WD"]
        in_dir = os.path.join(wd, "one_fasta_per_file")
        score_dict = defaultdict(dict)
        for fn in tqdm(glob(os.path.join(in_dir, "*align.txt"))):
            uniprot_id = os.path.basename(fn).replace("_align.txt", "")
            with open(fn, "r") as f:
                for line in f:
                    if line.startswith("target_name"):
                        target = line.split(":")[1].strip()
                    elif line.startswith("query_name"):
                        query = line.split(":")[1].strip()
                    elif line.startswith("optimal_alignment_score"):
                        score = int(line.split("\t")[0].split(": ")[1])

                        if query in score_dict[target]:
                            assert score == score_dict[target][query]
                        else:
                            score_dict[target][query] = score
        return score_dict

    def save(self):
        wd = self.config["DATA"]["WD"]
        with open(os.path.join(wd, "alignment_scores.pkl"), "wb") as f:
            pickle.dump(self.score_dict, f)



aligner = Aligner(config_fn)
parser = AlignmentParser(config_fn)
parser.save()
