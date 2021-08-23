partition="1080Ti,1080Ti_mlong,1080Ti_short,1080Ti_slong,2080Ti,2080Ti_mlong,CPUx16,CPUx40,M40x8,M40x8_mlong,M40x8_slong,P100,TitanXx8,TitanXx8_mlong,TitanXx8_slong,V100_DGX,V100x8"

for file in /mnt/scratch/boxiang/projects/cpi_contact/processed_data/unique_fasta/*fasta; do
    base=$(basename $file \.fasta)
    echo $base
    sbatch --ntasks=1 --cpus-per-task=8 --partition=$partition --job-name=$base --wrap "bash /mnt/scratch/boxiang/projects/cpi_contact/scripts/preprocess/AF2/run_alphafold2.sh $file /mnt/scratch/boxiang/projects/cpi_contact/processed_data/AF2/" --output=/mnt/scratch/boxiang/projects/cpi_contact/processed_data/AF2/${base}.out
done