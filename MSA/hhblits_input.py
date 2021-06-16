import sys
import pickle as pkl 
import os

in_fn = sys.argv[1]
out_dir = sys.argv[2]
if not os.path.exists(out_dir):
	os.makedirs(out_dir)

with open(in_fn, "rb") as f:
	interaction_dict = pkl.load(f)

n = 0
for k, v in interaction_dict.items():
	n += 1
	uniprot_id = v["uniprot_id"]
	pdb_id = k
	uniprot_seq = v["uniprot_seq"]
	seq_id = f"{pdb_id}_{uniprot_id}"
	with open(os.path.join(out_dir, f"{seq_id}.fasta"), "w") as f:
		f.write(f">{seq_id}\n")
		f.write(uniprot_seq)

sys.stderr.write(f"{n} sequences processed.\n")

