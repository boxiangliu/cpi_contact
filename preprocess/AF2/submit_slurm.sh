# partition="1080Ti,1080Ti_mlong,1080Ti_short,1080Ti_slong,2080Ti,2080Ti_mlong,CPUx16,CPUx40,M40x8,M40x8_mlong,M40x8_slong,P100,TitanXx8,TitanXx8_mlong,TitanXx8_slong,V100_DGX,V100x8"
partition="1080Ti_mlong,1080Ti_slong,2080Ti_mlong,CPUx16,M40x8_mlong,M40x8_slong,TitanXx8_mlong,TitanXx8_slong"

n=0
for file in /mnt/storage/idl-0/bio/boxiang/cpi_contact/processed_data/unique_fasta/*fasta; do
    base=$(basename $file .fasta)
    echo $base


    if [[ ! -d /mnt/storage/idl-0/bio/boxiang/cpi_contact/processed_data/AF2/$base ]]; then
        n=$((n+1))
        if [[ n -ge 1500 ]]; then 
            echo "Max number of tasks reached"
            echo $n
            break
        fi
        sbatch --ntasks=1 --cpus-per-task=8 --partition=$partition --job-name=$base --wrap "bash /mnt/scratch/boxiang/projects/alphafold/run_alphafold.sh $file /mnt/storage/idl-0/bio/boxiang/cpi_contact/processed_data/AF2/" --output=/mnt/storage/idl-0/bio/boxiang/cpi_contact/processed_data/AF2/${base}.out
    fi
done



# unfinished_file=$(grep -L "Final timing" *out | cut -f1 -d":" | sed "s/.out//" | tr "\n" " ")

# for file in $unfinished_file; do 
#     echo $file
#     rm -r $file
#     rm ${file}.out
# done