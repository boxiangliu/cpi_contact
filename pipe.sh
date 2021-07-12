# Create dataset: 
python dataset/get_dataset.py
bash dataset/run_plip.sh ../data/pdb_files/ ../data/plip_results/
python dataset/get_interaction.py
python dataset/get_alignment.py
python dataset/get_final_interaction.py


# Run hhblits:
bash MSA/run_hhblits.sh /mnt/scratch/boxiang/projects/cpi_contact/processed_data/MSA /mnt/scratch/boxiang/projects/cpi_contact/processed_data/MSA


ls /mnt/scratch/boxiang/projects/cpi_contact/data/MSA/*a3m > /mnt/scratch/boxiang/projects/cpi_contact/processed_data/MSA_features/fn_list
split -l 100 /mnt/scratch/boxiang/projects/cpi_contact/processed_data/MSA_features/fn_list /mnt/scratch/boxiang/projects/cpi_contact/processed_data/MSA_features/fn_list_

# queues=TitanXx8,TitanXx8_mlong,TitanXx8_slong,M40x8_slong,M40x8_mlong,M40x8,P100,1080Ti,2080Ti_mlong,2080Ti,CPUx40
queues=M40x8,M40x8_mlong,M40x8_slong

for f in /mnt/scratch/boxiang/projects/cpi_contact/processed_data/MSA_features/fn_list_*; do
    echo $f
    sleep 1
    sbatch -p $queues --job-name $(basename $f) --gres=gpu:1 --cpus-per-task 5 --wrap "python MSA/prepare_fasta.py --fn_list $f"
done