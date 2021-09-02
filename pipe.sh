##################
# Data Generation 
##################

# Download data:
python preprocess/get_dataset.py

# Use PLIP to get compound-protein interaction:
bash preprocess/run_plip.sh ../data/pdb_files/ ../data/plip_results/

# Parse PLIP compound-protein interaction result: 
python preprocess/get_interaction.py

# Align PDBBind and UniProt protein sequences:
python preprocess/get_alignment.py

# Use alignment to adjust interacting residues: 
python preprocess/get_final_interaction.py


# Generate MSA features:
# Generate fasta files:
python preprocess/MSA/hhblits_input.py ../data/results/out7_final_pairwise_interaction_dict ../processed_data/MSA/

# Run hhblits:
bash preprocess/MSA/run_hhblits.sh ../processed_data/MSA ../processed_data/MSA

# Extract MSA features using MSA transformer:
bash preprocess/MSA/extract_msa_features.sh

# Get alignment score:
bash preprocess/get_alignment_score.sh

# Final processing and clustering for train-dev-test split:
python preprocess/preprocess_and_clustering.py


############## 
# Alphafold 2
##############
python preprocess/AF2/unique_fasta.py ../processed_data/MSA/ ../processed_data/unique_fasta/
bash preprocess/AF2/submit_slurm.sh


############
# Modeling
############
python model/train.py IC50 new_compound 0.3

python src/train.py config/config.yaml /mnt/scratch/boxiang/projects/cpi_contact/data/preprocessed/models/test1/ --logtofile True
python src/train.py config/config.yaml /mnt/scratch/boxiang/projects/cpi_contact/data/preprocessed/models/new_cpd_thre_0.3_MSANet/ --logtofile True
python src/train.py config/config.yaml /mnt/scratch/boxiang/projects/cpi_contact/data/preprocessed/models/new_cpd_thre_0.4_MSANet/ --logtofile True
python src/train.py config/config.yaml /mnt/scratch/boxiang/projects/cpi_contact/data/preprocessed/models/new_cpd_thre_0.5_MSANet/ --logtofile True
python src/train.py config/config.yaml /mnt/scratch/boxiang/projects/cpi_contact/data/preprocessed/models/new_cpd_thre_0.6_MSANet/ --logtofile True

python src/train.py config/config.yaml /mnt/scratch/boxiang/projects/cpi_contact/data/preprocessed/models/new_cpd_thre_0.3_MONN/ --logtofile True
python src/train.py config/config.yaml /mnt/scratch/boxiang/projects/cpi_contact/data/preprocessed/models/new_cpd_thre_0.4_MONN/ --logtofile True
python src/train.py config/config.yaml /mnt/scratch/boxiang/projects/cpi_contact/data/preprocessed/models/new_cpd_thre_0.5_MONN/ --logtofile True
python src/train.py config/config.yaml /mnt/scratch/boxiang/projects/cpi_contact/data/preprocessed/models/new_cpd_thre_0.6_MONN/ --logtofile True