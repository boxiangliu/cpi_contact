import yaml
import sys

config_fn = sys.argv[1]

with open(config_fn, "r") as f:
    config = yaml.safe_load(f)

print(config)


