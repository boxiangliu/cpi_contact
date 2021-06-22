in_dir=$1
# in_dir = 
out_dir=$2
# out_dir=./plip_results/
mkdir -p $out_dir
cd $out_dir


run_plip(){
	f=$1
	base=$(basename $f .pdb)
	echo $base
	plipcmd.py -f $f -t --name ${base}_output
}
export -f run_plip

parallel -j 38 run_plip {} ::: $in_dir/????.pdb

