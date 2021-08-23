in_fasta=$1
out_dir=$2

python /mnt/scratch/boxiang/projects/alphafold/run_alphafold.py \
--fasta_paths=$in_fasta \
--output_dir=$out_dir \
--model_names=model_1 \
--uniref90_database_path=/mnt/storage/idl-0/bio/juneki/alphafold/data/uniref90/uniref90.fasta \
--mgnify_database_path=/mnt/storage/idl-0/bio/juneki/alphafold/data/mgnify/mgy_clusters.fa \
--pdb70_database_path=/mnt/storage/idl-0/bio/juneki/alphafold/data/pdb70/pdb70 \
--template_mmcif_dir=/mnt/storage/idl-0/bio/juneki/alphafold/data/pdb_mmcif/mmcif_files \
--max_template_date=2021-08-05 \
--obsolete_pdbs_path=/mnt/storage/idl-0/bio/juneki/alphafold/data/pdb_mmcif/obsolete.dat \
--small_bfd_database_path=/mnt/home/juneki/storage/alphafold/data/small_bfd/bfd-first_non_consensus_sequences.fasta \
--data_dir=/mnt/storage/idl-0/bio/juneki/alphafold/data/ \
--preset=reduced_dbs \
--hhblits_binary_path=hhblits \
--hhsearch_binary_path=hhsearch \
--jackhmmer_binary_path=jackhmmer