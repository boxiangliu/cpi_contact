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



def load_data(processed_dir, measure, setting, clu_thre, n_fold):
    # load data
    data_fn = os.path.join(processed_dir, "pdbbind_all_combined_input_" + measure)
    with open(data_fn, 'rb') as f:
        data_pack = pickle.load(f)
    cid_list = data_pack[7]
    pid_list = data_pack[8]
    n_sample = len(cid_list)
    
    # train-test split
    train_idx_list, valid_idx_list, test_idx_list = [], [], []
    print('setting:', setting)
    if setting == 'imputation':
        pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
        = split_train_test_clusters(measure, clu_thre, n_fold)
        for fold in range(n_fold):
            pair_train_valid, pair_test = pair_train_clusters[fold], pair_test_clusters[fold]
            pair_valid = np.random.choice(pair_train_valid, int(len(pair_train_valid)*0.125), replace=False)
            pair_train = set(pair_train_valid)-set(pair_valid)
            pair_valid = set(pair_valid)
            pair_test = set(pair_test)
            train_idx, valid_idx, test_idx = [], [], []
            for ele in range(n_sample):
                if 'c'+str(C_cluster_dict[cid_list[ele]])+'p'+str(P_cluster_dict[pid_list[ele]]) in pair_train:
                    train_idx.append(ele)
                elif 'c'+str(C_cluster_dict[cid_list[ele]])+'p'+str(P_cluster_dict[pid_list[ele]]) in pair_valid:
                    valid_idx.append(ele)
                elif 'c'+str(C_cluster_dict[cid_list[ele]])+'p'+str(P_cluster_dict[pid_list[ele]]) in pair_test:
                    test_idx.append(ele)
                else:
                    print('error')
            train_idx_list.append(train_idx)
            valid_idx_list.append(valid_idx)
            test_idx_list.append(test_idx)
            print('fold', fold, 'train ',len(train_idx),'test ',len(test_idx),'valid ',len(valid_idx))
            
    elif setting == 'new_protein':
        pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
        = split_train_test_clusters(measure, clu_thre, n_fold)
        for fold in range(n_fold):
            p_train_valid, p_test = p_train_clusters[fold], p_test_clusters[fold]
            p_valid = np.random.choice(p_train_valid, int(len(p_train_valid)*0.125), replace=False)
            p_train = set(p_train_valid)-set(p_valid)
            train_idx, valid_idx, test_idx = [], [], []
            for ele in range(n_sample): 
                if P_cluster_dict[pid_list[ele]] in p_train:
                    train_idx.append(ele)
                elif P_cluster_dict[pid_list[ele]] in p_valid:
                    valid_idx.append(ele)
                elif P_cluster_dict[pid_list[ele]] in p_test:
                    test_idx.append(ele)
                else:
                    print('error')
            train_idx_list.append(train_idx)
            valid_idx_list.append(valid_idx)
            test_idx_list.append(test_idx)
            print('fold', fold, 'train ',len(train_idx),'test ',len(test_idx),'valid ',len(valid_idx))
            
    elif setting == 'new_compound':
        pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
        = split_train_test_clusters(measure, clu_thre, n_fold)
        for fold in range(n_fold):
            c_train_valid, c_test = c_train_clusters[fold], c_test_clusters[fold]
            c_valid = np.random.choice(c_train_valid, int(len(c_train_valid)*0.125), replace=False)
            c_train = set(c_train_valid)-set(c_valid)
            train_idx, valid_idx, test_idx = [], [], []
            for ele in range(n_sample):
                if C_cluster_dict[cid_list[ele]] in c_train:
                    train_idx.append(ele)
                elif C_cluster_dict[cid_list[ele]] in c_valid:
                    valid_idx.append(ele)
                elif C_cluster_dict[cid_list[ele]] in c_test:
                    test_idx.append(ele)
                else:
                    print('error')
            train_idx_list.append(train_idx)
            valid_idx_list.append(valid_idx)
            test_idx_list.append(test_idx)
            print('fold', fold, 'train ',len(train_idx),'test ',len(test_idx),'valid ',len(valid_idx))
    
    elif setting == 'new_new':
        assert n_fold ** 0.5 == int(n_fold ** 0.5)
        pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
        = split_train_test_clusters(measure, clu_thre, int(n_fold ** 0.5))
        
        for fold_x in range(int(n_fold ** 0.5)):
            for fold_y in range(int(n_fold ** 0.5)):
                c_train_valid, p_train_valid = c_train_clusters[fold_x], p_train_clusters[fold_y]
                c_test, p_test = c_test_clusters[fold_x], p_test_clusters[fold_y]
                c_valid = np.random.choice(list(c_train_valid), int(len(c_train_valid)/3), replace=False)
                c_train = set(c_train_valid)-set(c_valid)
                p_valid = np.random.choice(list(p_train_valid), int(len(p_train_valid)/3), replace=False)
                p_train = set(p_train_valid)-set(p_valid)
                
                train_idx, valid_idx, test_idx = [], [], []
                for ele in range(n_sample):
                    if C_cluster_dict[cid_list[ele]] in c_train and P_cluster_dict[pid_list[ele]] in p_train:
                        train_idx.append(ele)
                    elif C_cluster_dict[cid_list[ele]] in c_valid and P_cluster_dict[pid_list[ele]] in p_valid:
                        valid_idx.append(ele)
                    elif C_cluster_dict[cid_list[ele]] in c_test and P_cluster_dict[pid_list[ele]] in p_test:
                        test_idx.append(ele)
                train_idx_list.append(train_idx)
                valid_idx_list.append(valid_idx)
                test_idx_list.append(test_idx)
                print('fold', fold_x*int(n_fold ** 0.5)+fold_y, 'train ',len(train_idx),'test ',len(test_idx),'valid ',len(valid_idx))
    
    return data_pack, train_idx_list, valid_idx_list, test_idx_list

def data_from_index(data_pack, idx_list):

    fa, fb, anb, bnb, nbs_mat, seq_input = [data_pack[i][idx_list] for i in range(6)]
    aff_label = data_pack[6][idx_list].astype(float).reshape(-1,1)
    pairwise_mask = data_pack[9][idx_list].astype(float).reshape(-1,1)
    pairwise_label = data_pack[10][idx_list]
    msa_feature = data_pack[11][idx_list]

    return [fa, fb, anb, bnb, nbs_mat, seq_input, aff_label, pairwise_mask, pairwise_label, msa_feature]

class CPIDataset(Dataset):
    def __init__(self, data):
        self.input_vertex, self.input_edge, \
        self.input_atom_adj, self.input_bond_adj, \
        self.input_num_nbs, self.input_seq, \
        self.affinity_label, self.pairwise_mask, \
        self.pairwise_label, self.msa_feature = data

        self._num_example = len(self.input_vertex)

    def __len__(self):
        return self._num_example

    def __getitem__(self, idx):
        return (self.input_vertex[idx], self.input_edge[idx], \
            self.input_atom_adj[idx], self.input_bond_adj[idx], \
            self.input_num_nbs[idx], self.input_seq[idx], \
            self.affinity_label[idx], self.pairwise_mask[idx], \
            self.pairwise_label[idx], self.msa_feature[idx])


def pack2D(arr_list):
    N = max([x.shape[0] for x in arr_list])
    M = max([x.shape[1] for x in arr_list])
    a = np.zeros((len(arr_list), N, M))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        m = arr.shape[1]
        a[i,0:n,0:m] = arr
    return a


def pack1D(arr_list):
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        a[i,0:n] = arr
    return a


def get_mask(arr_list):
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        a[i,:arr.shape[0]] = 1
    return a


#embedding selection function
def add_index(input_array, ebd_size):
    batch_size, n_vertex, n_nbs = np.shape(input_array)
    add_idx = np.array(list(range(0,(ebd_size)*batch_size,ebd_size))*(n_nbs*n_vertex))
    add_idx = np.transpose(add_idx.reshape(-1,batch_size))
    add_idx = add_idx.reshape(-1)
    new_array = input_array.reshape(-1)+add_idx
    return new_array

def collate_cpi(batch):

    vertex = []
    edge = []
    atom_adj = []
    bond_adj = []
    nbs = [] 
    sequence = []
    affinity_label = []
    pairwise_mask = []
    pairwise_label = []
    msa_feature = []

    for item in batch:
        vertex.append(item[0])
        edge.append(item[1])
        atom_adj.append(item[2])
        bond_adj.append(item[3])
        nbs.append(item[4])
        sequence.append(item[5])
        affinity_label.append(item[6])
        pairwise_mask.append(item[7])
        pairwise_label.append(item[8])
        msa_feature.append(item[9])

    vertex = np.array(vertex)
    edge = np.array(edge)
    atom_adj = np.array(atom_adj)
    bond_adj = np.array(bond_adj)
    nbs = np.array(nbs)
    sequence = np.array(sequence)
    affinity_label = np.array(affinity_label)
    pairwise_mask = np.array(pairwise_mask)
    pairwise_label = np.array(pairwise_label)
    msa_feature = np.array(msa_feature)

    vertex_mask = get_mask(vertex)
    vertex = pack1D(vertex)
    edge = pack1D(edge)
    atom_adj = pack2D(atom_adj)
    bond_adj = pack2D(bond_adj)
    nbs_mask = pack2D(nbs)
    
    #pad proteins and make masks
    seq_mask = get_mask(sequence)
    sequence = pack1D(sequence+1)
    msa_feature = pack2D(msa_feature)

    #add index
    atom_adj = add_index(atom_adj, np.shape(atom_adj)[1])
    bond_adj = add_index(bond_adj, np.shape(edge)[1])

    #convert to torch cuda data type
    vertex_mask = torch.FloatTensor(vertex_mask)
    vertex = torch.LongTensor(vertex)
    edge = torch.LongTensor(edge)
    atom_adj = torch.LongTensor(atom_adj)
    bond_adj = torch.LongTensor(bond_adj)
    nbs_mask = torch.FloatTensor(nbs_mask)
    
    seq_mask = torch.FloatTensor(seq_mask)
    sequence = torch.LongTensor(sequence)
    msa_feature = torch.FloatTensor(msa_feature)

    affinity_label = torch.FloatTensor(affinity_label)
    pairwise_mask = torch.FloatTensor(pairwise_mask)
    pairwise_label = torch.LongTensor(pairwise_label)


    return [vertex_mask, vertex, edge, atom_adj, \
            bond_adj, nbs_mask, seq_mask, sequence, \
            msa_feature, affinity_label, pairwise_mask, \
            pairwise_label]
