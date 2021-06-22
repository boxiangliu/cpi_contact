in_dir=$1
export out_dir=$2
mkdir -p $out_dir



run_plip(){
	f=$1
	base=$(basename $f .pdb)
	echo $base
	plipcmd.py -f $f -t --name $out_dir/${base}_output
}
export -f run_plip

parallel -j 38 run_plip {} ::: $in_dir/????.pdb

