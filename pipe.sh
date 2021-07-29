##################
# Data Generation 
##################

# Download data:
python data/get_dataset.py

# Use PLIP to get compound-protein interaction:
bash data/run_plip.sh ../data/pdb_files/ ../data/plip_results/

# Parse PLIP compound-protein interaction result: 
python dataset/get_interaction.py

# Align PDBBind and UniProt protein sequences:
python dataset/get_alignment.py

# Use alignment to adjust interacting residues: 
python dataset/get_final_interaction.py


# Generate MSA features:
# Generate fasta files:
python data/MSA/hhblits_input.py ../data/results/out7_final_pairwise_interaction_dict ../processed_data/MSA/

# Run hhblits:
bash data/MSA/run_hhblits.sh ../processed_data/MSA ../processed_data/MSA

# Extract MSA features using MSA transformer:
bash data/MSA/extract_msa_features.sh

# Get alignment score:
bash get_alignment_score.sh

# Final processing and clustering for train-dev-test split:
python preprocess_and_clustering.py


python model/train.py IC50 new_compound 0.3