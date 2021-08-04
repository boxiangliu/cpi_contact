import yaml
from easydict import EasyDict as edict
import pickle
import numpy as np
import os
import sys
import torch
from torch.autograd import Variable
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
sys.path.append(os.getcwd() + "/model/")
from model import Net
import math
from torch.utils.data import Dataset


elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
CFG = "config/config.yaml"

class Trainer(object):
    def __init__(self, cfg， train_data, valid_data, test_data):
        self.setup(cfg)
        self.init_model()

    def setup(self, cfg):
        with open(cfg) as f:
            self.cfg = edict(yaml.full_load(f))

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
        self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def init_data(self):
        # train_cfg = self.cfg.TRAIN
        # data_pack, train_idx_list, valid_idx_list, test_idx_list = \
        #     load_data(processed_dir=train_cfg.PROCESSED,
        #               measure=train_cfg.MEASURE,
        #               setting=train_cfg.SETTING,
        #               clu_thre=train_cfg.CLU_THRE,
        #               n_fold=train_cfg.N_FOLD)


    def init_logging(self):
        pass

    def train_step(self):
        pass

    def dev_epoch(self):
        pass

    def save_model(self):
        pass

trainer = Trainer(CFG)



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



