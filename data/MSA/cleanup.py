from glob import glob
import torch
from utils import config_fn
import yaml 
import os
from tqdm import tqdm

with open(config_fn, "r") as f:
    config = yaml.safe_load(f)

msa_feature_dir = config["DATA"]["MSA_FEATURES"]
n_total, n_uniq = 0, 0
for fn in tqdm(glob(os.path.join(msa_feature_dir, "*.pt"))):
    n_total += 1
    pt = torch.load(fn, map_location=torch.device("cpu"))
    pt["id"] = pt["id"].split("_")[1]
    out_fn = os.path.join(msa_feature_dir, pt["id"] + ".pt")
    if os.path.exists(out_fn):
        continue
    n_uniq += 1
    torch.save(pt, out_fn)

sys.stderr.write(f"Total number of pt files: {n_total}\n")
sys.stderr.write(f"Number of unique pt files: {n_total}\n")
