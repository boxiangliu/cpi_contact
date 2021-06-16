in_dir=$1
out_dir=$2
queues=TitanXx8,TitanXx8_mlong,TitanXx8_slong,M40x8_slong,M40x8_mlong,M40x8,P100,1080Ti,2080Ti_mlong,2080Ti,CPUx40

touch $out_dir/command_list
for f in $in_dir/*fasta; do
	echo $f
	base=$(basename $f .fasta)
	echo "hhblits -i $f -o $out_dir/${base}.hhr -oa3m $out_dir/${base}.a3m -d /mnt/home/vincent/protein_folding/data/UniRef30_2020_02" >> $out_dir/command_list 
done 

split -n l/800 $out_dir/command_list $out_dir/run_

for f in $out_dir/run_*; do
	echo $f
	sleep 1
	sbatch -p $queues --job-name $(basename $f) --wrap "bash $f" 
done
