import sys
import os
import argparse
from data.dataset import load_data, data_from_index
from engine.trainer import Trainer
import yaml
from easydict import EasyDict as edict

parser = argparse.ArgumentParser(description="Train model")
parser.add_argument("cfg_file",
                    default=None,
                    metavar="CFG_FILE",
                    type=str,
                    help="Model configuration file")
parser.add_argument("save_path",
                    default=None,
                    metavar="SAVE_PATH",
                    type=str,
                    help="Path to the saved models")
parser.add_argument("--num_workers",
                    default=1,
                    type=int,
                    help="Number of workers for each data loader")
parser.add_argument("--device_ids",
                    default="0",
                    type=str,
                    help="GPU indices comma separated, e.g. '0,1'")
parser.add_argument("--logtofile",
                    default=False,
                    type=bool,
                    help="Save log in <save_path>/log.txt if True")


class DummyArgs(object):
    def __init__(self):
        self.cfg_file = "config/config.yaml"
        self.save_path = "/mnt/scratch/boxiang/projects/cpi_contact/data/preprocessed/models/test/"

args = DummyArgs()

def run(args):
    with open(args.cfg_file) as f:
        cfg = edict(yaml.full_load(f))


    data_pack, train_idx_list, valid_idx_list, test_idx_list = \
        load_data(processed_dir=cfg.TRAIN.PROCESSED,
              measure=cfg.TRAIN.MEASURE,
              setting=cfg.TRAIN.SETTING,
              clu_thre=cfg.TRAIN.CLU_THRE,
              n_fold=cfg.TRAIN.N_FOLD)

    fold = 0
    train_data = data_from_index(data_pack, train_idx_list[fold])
    valid_data = data_from_index(data_pack, valid_idx_list[fold])
    test_data = data_from_index(data_pack, test_idx_list[fold])

    trainer = Trainer(args, train_data, valid_data, test_data)

    epoch_steps_train = len(trainer.train_loader)
    total_steps_train = epoch_steps_train * cfg.TRAIN.EPOCH
    start_step = trainer.summary["step"]

    for step in range(start_step, total_steps_train):
        trainer.train_step()

        if (step + 1) % cfg.TRAIN.LOG_EVERY == 0:
            trainer.logging(mode="Train")
            trainer.write_summary(mode="Train")
            trainer.reset_log()

        if (step + 1) % cfg.TRAIN.DEV_EVERY == 0:
            trainer.dev_epoch()
            trainer.logging(mode="Dev")
            trainer.write_summary(mode="Dev")
            trainer.save_model(mode="Train")
            trainer.save_model(mode="Dev")

    trainer.close()

def main():
    args = parser.parse_args()
    if args.verbose is True:
        print("Arguments:")
        print(args)
    run(args)


