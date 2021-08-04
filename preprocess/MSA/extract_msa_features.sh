wd=/mnt/scratch/boxiang/projects/cpi_contact

# Generate input file lists:
ls ${wd}/processed_data/MSA/*a3m > ${wd}/processed_data/MSA_features/fn_list
split -l 100 ${wd}/processed_data/MSA_features/fn_list ${wd}/processed_data/MSA_features/fn_list_

# Extract MSA features:
# queues=TitanXx8,TitanXx8_mlong,TitanXx8_slong,M40x8_slong,M40x8_mlong,M40x8,P100,1080Ti,2080Ti_mlong,2080Ti,CPUx40
queues=M40x8,M40x8_mlong,M40x8_slong
for f in ${wd}/processed_data/MSA_features/fn_list_*; do
    echo $f
    sleep 1
    sbatch -p $queues --job-name $(basename $f) --gres=gpu:1 --cpus-per-task 5 --wrap "python data/MSA/extract_msa_features.py --fn_list $f"
done
