import os
import pickle 
import numpy as np
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset, DataLoader

def split_train_test_clusters(processed_dir, measure, clu_thre, n_fold):
    # load cluster dict
    cluster_path = processed_dir
    with open(cluster_path+measure+'_compound_cluster_dict_'+str(clu_thre), 'rb') as f:
        C_cluster_dict = pickle.load(f)
    with open(cluster_path+measure+'_protein_cluster_dict_'+str(clu_thre), 'rb') as f:
        P_cluster_dict = pickle.load(f)
    
    C_cluster_set = set(list(C_cluster_dict.values()))
    P_cluster_set = set(list(P_cluster_dict.values()))
    C_cluster_list = np.array(list(C_cluster_set))
    P_cluster_list = np.array(list(P_cluster_set))
    np.random.shuffle(C_cluster_list)
    np.random.shuffle(P_cluster_list)
    # n-fold split
    # c_kf = KFold(len(C_cluster_list), n_fold, shuffle=True)
    # p_kf = KFold(len(P_cluster_list), n_fold, shuffle=True)
    c_kf = KFold(n_fold,shuffle=True)
    p_kf = KFold(n_fold,shuffle=True)
    c_train_clusters, c_test_clusters = [], []
    for train_idx, test_idx in c_kf.split(C_cluster_list):
        c_train_clusters.append(C_cluster_list[train_idx])
        c_test_clusters.append(C_cluster_list[test_idx])
    p_train_clusters, p_test_clusters = [], []
    for train_idx, test_idx in p_kf.split(P_cluster_list):
        p_train_clusters.append(P_cluster_list[train_idx])
        p_test_clusters.append(P_cluster_list[test_idx])
    
    
    pair_kf = KFold(n_fold,shuffle=True)
    pair_list = []
    for i_c in C_cluster_list:
        for i_p in P_cluster_list:
            pair_list.append('c'+str(i_c)+'p'+str(i_p))
    pair_list = np.array(pair_list)
    np.random.shuffle(pair_list)
    # pair_kf = KFold(len(pair_list), n_fold, shuffle=True)
    pair_train_clusters, pair_test_clusters = [], []
    for train_idx, test_idx in pair_kf.split(pair_list):
        pair_train_clusters.append(pair_list[train_idx])
        pair_test_clusters.append(pair_list[test_idx])
    
    return pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict


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
        = split_train_test_clusters(processed_dir, measure, clu_thre, n_fold)
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
        = split_train_test_clusters(processed_dir, measure, clu_thre, n_fold)
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
        = split_train_test_clusters(processed_dir, measure, clu_thre, n_fold)
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
        = split_train_test_clusters(processed_dir, measure, clu_thre, int(n_fold ** 0.5))
        
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


def pad_label_2d(label, vertex, sequence):
    dim1 = vertex.size(1)
    dim2 = sequence.size(1)
    a = np.zeros((len(label), dim1, dim2))
    for i, arr in enumerate(label):
        a[i, :arr.shape[0], :arr.shape[1]] = arr
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

    vertex = np.array(vertex, dtype=object)
    edge = np.array(edge, dtype=object)
    atom_adj = np.array(atom_adj, dtype=object)
    bond_adj = np.array(bond_adj, dtype=object)
    nbs = np.array(nbs, dtype=object)
    sequence = np.array(sequence, dtype=object)
    msa_feature = np.array(msa_feature, dtype=object)

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
    pairwise_label = torch.FloatTensor(pad_label_2d(pairwise_label, vertex, sequence))


    return [vertex_mask, vertex, edge, atom_adj, \
            bond_adj, nbs_mask, seq_mask, sequence, \
            msa_feature, affinity_label, pairwise_mask, \
            pairwise_label]


data_pack, train_idx_list, valid_idx_list, test_idx_list = \
    load_data(processed_dir="/mnt/scratch/boxiang/projects/cpi_contact/data/preprocessed/",
              measure="IC50",
              setting="new_compound",
              clu_thre=0.3,
              n_fold=5)

# fold = 0
# train_data = data_from_index(data_pack, train_idx_list[fold])
# valid_data = data_from_index(data_pack, valid_idx_list[fold])
# test_data = data_from_index(data_pack, test_idx_list[fold])

# train_data = CPIDataset(train_data)

# data_loader = DataLoader(train_data, batch_size=16, collate_fn=collate_cpi)
# data_loader = iter(data_loader)
# batch = next(data_loader)