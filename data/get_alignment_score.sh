# Perform pairwise alignment across all PDBBind sequences:
sw_dir=dataset/smith-waterman-src/
fasta_fn=../data/results/out6.1_target_uniprot_pdb.fasta
out_fn=../data/results/out6.1_target_uniprot_pdb_align.txt
python2 ${sw_dir}/pyssw.py -l $sw_dir -c -p $fasta_fn $fasta_fn > $out_fn

# Parse alignment score and save to disk:
python data/get_alignment_score.py