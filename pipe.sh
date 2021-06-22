# Create dataset: 
python dataset/get_dataset.py 
bash dataset/run_plip.sh ../data/pdb_files/ ../data/plip_results/

# Run hhblits:
bash MSA/run_hhblits.sh /mnt/scratch/boxiang/projects/cpi_contact/processed_data/MSA /mnt/scratch/boxiang/projects/cpi_contact/processed_data/MSA

