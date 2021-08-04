import torch
import torch.nn as nn
from tape.tape.models.modeling_utils import ProteinConfig

URL_PREFIX = "https://s3.amazonaws.com/proteindata/pytorch-models/"
TRROSETTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'xaa': URL_PREFIX + "trRosetta-xaa-pytorch_model.bin",
    'xab': URL_PREFIX + "trRosetta-xab-pytorch_model.bin",
    'xac': URL_PREFIX + "trRosetta-xac-pytorch_model.bin",
    'xad': URL_PREFIX + "trRosetta-xad-pytorch_model.bin",
    'xae': URL_PREFIX + "trRosetta-xae-pytorch_model.bin",
}
TRROSETTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'xaa': URL_PREFIX + "trRosetta-xaa-config.json",
    'xab': URL_PREFIX + "trRosetta-xab-config.json",
    'xac': URL_PREFIX + "trRosetta-xac-config.json",
    'xad': URL_PREFIX + "trRosetta-xad-config.json",
    'xae': URL_PREFIX + "trRosetta-xae-config.json",
}


class TRRosettaConfig(ProteinConfig):

    pretrained_config_archive_map = TRROSETTA_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 num_features: int = 64,
                 kernel_size: int = 3,
                 num_layers: int = 61,
                 dropout: float = 0.15,
                 msa_cutoff: float = 0.8,
                 penalty_coeff: float = 4.5,
                 initializer_range: float = 0.02,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.msa_cutoff = msa_cutoff
        self.penalty_coeff = penalty_coeff
        self.initializer_range = initializer_range



class MSAFeatureExtractor(nn.Module):

    def __init__(self, config: TRRosettaConfig):
        super().__init__()
        self.msa_cutoff = config.msa_cutoff
        self.penalty_coeff = config.penalty_coeff

    def forward(self, msa1hot):
        # Convert to float, then potentially back to half
        # These transforms aren't well suited to half-precision
        initial_type = msa1hot.dtype

        msa1hot = msa1hot.float()
        seqlen = msa1hot.size(2)

        weights = self.reweight(msa1hot)
        features_1d = self.extract_features_1d(msa1hot, weights)
        features_2d = self.extract_features_2d(msa1hot, weights)

        left = features_1d.unsqueeze(2).repeat(1, 1, seqlen, 1)
        right = features_1d.unsqueeze(1).repeat(1, seqlen, 1, 1)
        features = torch.cat((left, right, features_2d), -1)
        features = features.type(initial_type)
        features = features.permute(0, 3, 1, 2)
        features = features.contiguous()
        return features

    def reweight(self, msa1hot, eps=1e-9):
        # Reweight
        seqlen = msa1hot.size(2)
        id_min = seqlen * self.msa_cutoff
        id_mtx = torch.stack([torch.tensordot(el, el, [[1, 2], [1, 2]]) for el in msa1hot], 0)
        id_mask = id_mtx > id_min
        weights = 1.0 / (id_mask.type_as(msa1hot).sum(-1) + eps)
        return weights

    def extract_features_1d(self, msa1hot, weights):
        # 1D Features
        f1d_seq = msa1hot[:, 0, :, :20]
        batch_size = msa1hot.size(0)
        seqlen = msa1hot.size(2)

        # msa2pssm
        beff = weights.sum()
        f_i = (weights[:, :, None, None] * msa1hot).sum(1) / beff + 1e-9
        h_i = (-f_i * f_i.log()).sum(2, keepdims=True)
        f1d_pssm = torch.cat((f_i, h_i), dim=2)
        f1d = torch.cat((f1d_seq, f1d_pssm), dim=2)
        f1d = f1d.view(batch_size, seqlen, 42)
        return f1d

    def extract_features_2d(self, msa1hot, weights):
        # 2D Features
        batch_size = msa1hot.size(0)
        num_alignments = msa1hot.size(1)
        seqlen = msa1hot.size(2)
        num_symbols = 21

        if num_alignments == 1:
            # No alignments, predict from sequence alone
            f2d_dca = torch.zeros(
                batch_size, seqlen, seqlen, 442,
                dtype=torch.float,
                device=msa1hot.device)
            return f2d_dca

        # compute fast_dca
        # covariance
        x = msa1hot.view(batch_size, num_alignments, seqlen * num_symbols)
        num_points = weights.sum(1) - weights.mean(1).sqrt()
        mean = (x * weights.unsqueeze(2)).sum(1, keepdims=True) / num_points[:, None, None]
        x = (x - mean) * weights[:, :, None].sqrt()
        cov = torch.matmul(x.transpose(-1, -2), x) / num_points[:, None, None]

        # inverse covariance
        reg = torch.eye(seqlen * num_symbols,
                        device=weights.device,
                        dtype=weights.dtype)[None]
        reg = reg * self.penalty_coeff / weights.sum(1, keepdims=True).sqrt().unsqueeze(2)
        cov_reg = cov + reg
        inv_cov = torch.stack([torch.inverse(cr) for cr in cov_reg.unbind(0)], 0)

        x1 = inv_cov.view(batch_size, seqlen, num_symbols, seqlen, num_symbols)
        x2 = x1.permute(0, 1, 3, 2, 4)
        features = x2.reshape(batch_size, seqlen, seqlen, num_symbols * num_symbols)

        x3 = (x1[:, :, :-1, :, :-1] ** 2).sum((2, 4)).sqrt() * (
            1 - torch.eye(seqlen, device=weights.device, dtype=weights.dtype)[None])
        apc = x3.sum(1, keepdims=True) * x3.sum(2, keepdims=True) / x3.sum(
            (1, 2), keepdims=True)
        contacts = (x3 - apc) * (1 - torch.eye(
            seqlen, device=x3.device, dtype=x3.dtype).unsqueeze(0))

        f2d_dca = torch.cat([features, contacts[:, :, :, None]], axis=3)
        return f2d_dca

    @property
    def feature_size(self) -> int:
        return 526

import ipdb; ipdb.set_trace()
trRosetta_config = TRRosettaConfig()
feature_extractor = MSAFeatureExtractor(trRosetta_config)

onehot = torch.load("T1001.pt")
feature = feature_extractor(onehot.unsqueeze(0))
