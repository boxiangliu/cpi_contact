# Create dataset: 
python dataset/get_dataset.py
bash dataset/run_plip.sh ../data/pdb_files/ ../data/plip_results/
python dataset/get_interaction.py
python dataset/get_alignment.py
python dataset/get_final_interaction.py

################
# Generate MSA
################
# Run hhblits:
bash MSA/run_hhblits.sh /mnt/scratch/boxiang/projects/cpi_contact/processed_data/MSA /mnt/scratch/boxiang/projects/cpi_contact/processed_data/MSA

# Generate input file lists:
ls /mnt/scratch/boxiang/projects/cpi_contact/data/MSA/*a3m > /mnt/scratch/boxiang/projects/cpi_contact/processed_data/MSA_features/fn_list
split -l 100 /mnt/scratch/boxiang/projects/cpi_contact/processed_data/MSA_features/fn_list /mnt/scratch/boxiang/projects/cpi_contact/processed_data/MSA_features/fn_list_

# Extract MSA features:
# queues=TitanXx8,TitanXx8_mlong,TitanXx8_slong,M40x8_slong,M40x8_mlong,M40x8,P100,1080Ti,2080Ti_mlong,2080Ti,CPUx40
queues=M40x8,M40x8_mlong,M40x8_slong
for f in /mnt/scratch/boxiang/projects/cpi_contact/processed_data/MSA_features/fn_list_*; do
    echo $f
    sleep 1
    sbatch -p $queues --job-name $(basename $f) --gres=gpu:1 --cpus-per-task 5 --wrap "python MSA/extract_msa_features.py --fn_list $f"
done


# Perform pairwise alignment across all PDBBind sequences:
sw_dir=dataset/smith-waterman-src/
fasta_fn=../data/results/out6.1_target_uniprot_pdb.fasta
out_fn=../data/results/out6.1_target_uniprot_pdb_align.txt
python2 ${sw_dir}/pyssw.py -l $sw_dir -c -p $fasta_fn $fasta_fn > $out_fn