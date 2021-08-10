import yaml
from easydict import EasyDict as edict
import pickle
import numpy as np
import os
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
sys.path.append(os.getcwd() + "/model/")
from model import Net
import math
from torch.utils.data import Dataset
import time
from tensorboardX import SummaryWriter
from torch import optim
from data.dataset import CPIDataset

elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
CFG = "config/config.yaml"

class Trainer(object):
    def __init__(self, args, train_data, valid_data, test_data):
        self.args = args
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        self.setup()
        self.init_model()
        self.init_data()
        self.init_log()

    def setup(self):
        with open(self.args.cfg_file) as f:
            self.cfg = edict(yaml.full_load(f))

        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)

        self.device = torch.device("cuda")
        self.summary_writer = SummaryWriter(self.args.save_path)

    def init_model(self):
        train_cfg = self.cfg.TRAIN

        blosum_dict = load_blosum62(train_cfg.BLOSUM62)
        init_A, init_B, init_W = loading_emb(
            processed_dir=train_cfg.PROCESSED, 
            measure=train_cfg.MEASURE, 
            blosum_dict=blosum_dict)

        params = [train_cfg.GNN_DEPTH, 
                  train_cfg.CNN_DEPTH, 
                  train_cfg.DMA_DEPTH, 
                  train_cfg.K_HEAD, 
                  train_cfg.KERNEL_SIZE, 
                  train_cfg.HIDDEN_SIZE_1, 
                  train_cfg.HIDDEN_SIZE_2, 
                  train_cfg.HIDDEN_SIZE_3]

        net = Net(init_A, init_B, init_W, params).cuda()
        net.apply(weights_init)
        total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        sys.stderr.write("Total parameters: {}".format(total_params))
        self.net = net

        self.criterion1 = nn.MSELoss()
        self.criterion2 = Masked_BCELoss()

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0005, weight_decay=0, amsgrad=True)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

    def init_data(self):
        self.train_dataset = CPIDataset(self.train_data)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.cfg.TRAIN.BATCH_SIZE, collate_fn=collate_cpi)
        self.train_iter = iter(train_loader)

        self.valid_dataset = CPIDataset(self.valid_data)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.cfg.TRAIN.BATCH_SIZE, collate_fn=collate_cpi)

    def init_log(self):
        self.summary = {
            "step": 0,
            "log_step": 0,
            "epoch": 1,
            "loss_sum": 0.0,
            "loss_pairwise_sum": 0.0,
            "loss_aff_sum": 0.0,
            "loss_dev": 0.0,
            "loss_pairwise_dev": 0.0,
            "loss_aff_dev": 0.0,
            "loss_dev_best": float("inf"),
            "loss_pairwise_dev_best": float("inf"),
            "loss_aff_dev_best": float("inf")
        }
        self.time_stamp = time.time()

    def reset_log(self):
        self.summary["log_step"] = 0
        self.summary["loss_sum"] = 0.0
        self.summary["loss_pairwise_sum"] = 0.0
        self.summary["loss_aff_sum"] = 0.0

    def logging(self, mode="Train"):
        time_elapsed = time.time() - self.time_stamp
        self.time_stamp = time.time()

        if mode == "Train":
            log_step = self.summary["log_step"]
            loss = self.summary["loss_sum"] / log_step
            loss_pairwise = self.summary["loss_pairwise_sum"] / log_step
            loss_aff = self.summary["loss_aff_sum"] / log_step

            logging.info(
                "{}, Train, Epoch: {}, Step: {}, Loss: {:.3f}, "
                "PairLoss: {:.3f}, AffLoss: {:.3f}, Runtime: {:.2f} s"
                .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                        self.summary["epoch"], self.summary["step"],
                        loss, loss_pairwise, loss_aff, time_elapsed))

        elif mode == "Dev":
            logging.info(
                "{}, Dev, Epoch: {}, Step: {}, Loss: {:.3f}, "
                "PairLoss: {:.3f}, AffLoss: {:.3f}, Runtime: {:.2f} s"
                .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                        self.summary["epoch"], self.summary["step"],
                        self.summary["loss_dev"], self.summary["loss_pairwise_dev"],
                        self.summary["loss_aff_dev"], time_elapsed))

    def train_step(self):
        try:
            vertex_mask, vertex, edge, atom_adj, \
            bond_adj, nbs_mask, seq_mask, sequence, \
            msa_feature, affinity_label, pairwise_mask, \
            pairwise_label = next(self.train_iter)

        except StopIteration:
            self.summary['epoch'] += 1
            self.train_iter = iter(self.train_loader)
            vertex_mask, vertex, edge, atom_adj, \
            bond_adj, nbs_mask, seq_mask, sequence, \
            msa_feature, affinity_label, pairwise_mask, \
            pairwise_label = next(self.train_iter)

        vertex_mask = vertex_mask.to(self.device)
        vertex = vertex.to(self.device)
        edge = edge.to(self.device)
        atom_adj = atom_adj.to(self.device)
        bond_adj = bond_adj.to(self.device)
        nbs_mask = nbs_mask.to(self.device)
        seq_mask = seq_mask.to(self.device)
        sequence = sequence.to(self.device)
        msa_feature = msa_feature.to(self.device)
        affinity_label = affinity_label.to(self.device)
        pairwise_mask = pairwise_mask.to(self.device)
        pairwise_label = pairwise_label.to(self.device)


        affinity_pred, pairwise_pred = self.net(
            vertex_mask, vertex, edge, \
            atom_adj, bond_adj, nbs_mask, \
            seq_mask, sequence, msa_feature)

        loss_aff = self.criterion1(affinity_pred, affinity_label)
        loss_pairwise = self.criterion2(pairwise_pred, pairwise_label, pairwise_mask, vertex_mask, seq_mask)
        loss = loss_aff + 0.1*loss_pairwise

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()

        self.summary["loss_sum"] += loss_aff.item()
        self.summary["loss_pairwise_sum"] += loss_pairwise.item()
        self.summary["loss_aff_sum"] += loss_aff.item()
        self.summary["step"] += 1
        self.summary["log_step"] += 1

    def dev_epoch(self):
        self.time_stamp = time.time()
        torch.set_grad_enabled(False)
        self.net.eval()

        steps = len(self.valid_loader)
        valid_iter = iter(self.valid_loader)
        loss_sum = 0.0
        loss_pairwise_sum = 0.0
        loss_aff_sum = 0.0

        for step in range(steps):
            (vertex_mask, vertex, edge, atom_adj, 
             bond_adj, nbs_mask, seq_mask, sequence, 
             msa_feature, affinity_label, pairwise_mask, 
             pairwise_label) = next(self.train_iter)

            vertex_mask = vertex_mask.to(self.device)
            vertex = vertex.to(self.device)
            edge = edge.to(self.device)
            atom_adj = atom_adj.to(self.device)
            bond_adj = bond_adj.to(self.device)
            nbs_mask = nbs_mask.to(self.device)
            seq_mask = seq_mask.to(self.device)
            sequence = sequence.to(self.device)
            msa_feature = msa_feature.to(self.device)
            affinity_label = affinity_label.to(self.device)
            pairwise_mask = pairwise_mask.to(self.device)
            pairwise_label = pairwise_label.to(self.device)

            affinity_pred, pairwise_pred = self.net(
                vertex_mask, vertex, edge, \
                atom_adj, bond_adj, nbs_mask, \
                seq_mask, sequence, msa_feature)

            loss_aff = self.criterion1(affinity_pred, affinity_label)
            loss_pairwise = self.criterion2(pairwise_pred, pairwise_label, pairwise_mask, vertex_mask, seq_mask)
            loss = loss_aff + 0.1*loss_pairwise

            loss_sum += loss.item()
            loss_pairwise_sum += loss.item()
            loss_aff_sum += loss.item()

        self.summary["loss_dev"] = loss_sum / steps
        self.summary["loss_pairwise_dev"] = loss_pairwise_sum / steps
        self.summary["loss_aff_dev"] = loss_aff_sum / steps

        torch.set_grad_enabled(True)
        self.net.train()

    def write_summary(self, mode="Train"):
        if mode == "Train":
            self.summary_writer.add_scalar(
                "Train/loss",
                self.summary["loss_sum"] / self.summary["log_step"],
                self.summary["step"])
            self.summary_writer.add_scalar(
                "Train/loss_pairwise",
                self.summary["loss_pairwise_sum"] / self.summary["log_step"],
                self.summary["step"])
            self.summary_writer.add_scalar(
                "Train/loss_aff",
                self.summary["loss_aff_sum"] / self.summary["log_step"],
                self.summary["step"])

        elif mode == "Dev":
            self.summary_writer.add_scalar(
                "Dev/loss",
                self.summary["loss_dev"],
                self.summary["step"])
            self.summary_writer.add_scalar(
                "Dev/loss_pairwise",
                self.summary["loss_pairwise_dev"],
                self.summary["step"])
            self.summary_writer.add_scalar(
                "Dev/loss_aff",
                self.summary["loss_aff_dev"],
                self.summary["step"])


    def save_model(self, mode="Train"):
        if mode == "Train":
            torch.save(
                {
                    "epoch": self.summary["epoch"],
                    "step": self.summary["step"],
                    "loss_dev_best": self.summary["loss_dev_best"],
                    "loss_pairwise_dev_best": self.summary["loss_pairwise_dev_best"],
                    "loss_aff_dev_best": self.summary["loss_aff_dev_best"]
                },
                os.path.join(self.args.save_path, "train.ckpt"))
        elif model == "Dev":
            save_best = False
            if self.summary["loss_dev"].mean() < self.summary["loss_dev_best"]:
                self.summary["loss_dev_best"] = self.summary["loss_dev"].mean()
                self.summary["loss_pairwise_dev_best"] = self.summary["loss_pairwise_dev"].mean()
                self.summary["loss_aff_dev_best"] = self.summary["loss_aff_dev"].mean()
                save_best = True

            if save_best:
                torch.save(
                    {
                        "epoch": self.summary["epoch"],
                        "step": self.summary["step"],
                        "loss_dev_best": self.summary["loss_dev_best"],
                        "loss_pairwise_dev_best": self.summary["loss_pairwise_dev_best"],
                        "loss_aff_dev_best": self.summary["loss_aff_dev_best"]
                    },
                    os.path.join(self.args.save_path, "best.ckpt"))

                logging.info(
                    "{}, Best, Epoch: {}, Step: {}, Loss: {:.3f}, "
                    "PairLoss: {:.3f}, AffLoss: {:.3f}"
                    .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                            self.summary["epoch"], self.summary["step"],
                            self.summary["loss_dev"], self.summary["loss_pairwise_dev"],
                            self.summary["loss_aff_dev"]))

    def close(self):
        self.summary_writer.close()


def loading_emb(processed_dir, measure, blosum_dict):
    fn = os.path.join(processed_dir, "pdbbind_all_atom_dict_{}".format(measure))
    with open(fn, "rb") as f:
        atom_dict = pickle.load(f)

    fn = os.path.join(processed_dir, "pdbbind_all_bond_dict_{}".format(measure))
    with open(fn, "rb") as f:
        bond_dict = pickle.load(f)

    fn = os.path.join(processed_dir, "pdbbind_all_word_dict_{}".format(measure))
    with open(fn, "rb") as f:
        word_dict = pickle.load(f)

    sys.stderr.write("Atom dict size: {}\n".format(len(atom_dict)))
    sys.stderr.write("Bond dict size: {}\n".format(len(bond_dict)))
    sys.stderr.write("Word dict size: {}\n".format(len(word_dict)))

    init_atom_features = np.zeros((len(atom_dict), atom_fdim))
    init_bond_features = np.zeros((len(bond_dict), bond_fdim))
    init_word_features = np.zeros((len(word_dict), 20))
    for key,value in atom_dict.items():
        init_atom_features[value] = np.array(list(map(int, key)))
    
    for key,value in bond_dict.items():
        init_bond_features[value] = np.array(list(map(int, key)))

    for key, value in word_dict.items():
        if key not in blosum_dict:
            continue 
        init_word_features[value] = blosum_dict[key]
    init_word_features = torch.cat([torch.zeros(1,20), torch.FloatTensor(init_word_features)], dim=0)

    for features in [init_atom_features, init_bond_features, init_word_features]:
        features = Variable(torch.FloatTensor(features)).cuda()

    return init_atom_features, init_bond_features, init_word_features

def load_blosum62(fn):
    blosum_dict = {}
    with open(fn) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            split_line = line.strip("\n").split()
            blosum_dict[split_line[0]] = np.array(split_line[1:], dtype=float)
    return blosum_dict


#Model parameter intializer
def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m,nn.Linear):
        nn.init.normal_(m.weight.data, mean=0, std=min(1.0 / math.sqrt(m.weight.data.shape[-1]), 0.1))
        nn.init.constant_(m.bias, 0)

class Masked_BCELoss(nn.Module):
    def __init__(self):
        super(Masked_BCELoss, self).__init__()
        self.criterion = nn.BCELoss(reduce=False)

    def forward(self, pred, label, pairwise_mask, vertex_mask, seq_mask):
        batch_size = pred.size(0)
        loss_all = self.criterion(pred, label)
        loss_mask = torch.matmul(vertex_mask.view(batch_size,-1,1), seq_mask.view(batch_size,1,-1))*pairwise_mask.view(-1, 1, 1)
        loss = torch.sum(loss_all*loss_mask) / torch.sum(pairwise_mask).clamp(min=1e-10)
        return loss



