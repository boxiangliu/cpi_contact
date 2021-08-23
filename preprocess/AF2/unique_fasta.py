import sys
import glob
from shutil import copyfile
import os

msa_dir = sys.argv[1]
uniq_dir = sys.argv[2]
if os.path.exists(uniq_dir):
    os.mkdirs(uniq_dir)

uniprot_list = []
for file in glob.glob(msa_dir + "/*.fasta"):
    uniprot_id = os.path.basename(file).replace(".fasta", "").split("_")[1]
    if uniprot_id not in uniprot_list:
        uniprot_list.append(uniprot_id)
        copyfile(file, os.path.join(uniq_dir, "{}.fasta".format(uniprot_id)))


