#!/usr/bin/env python
import torch 
import ase
from e3_layers.utils import build
from e3_layers import configs
from e3_layers.data import Batch, computeEdgeIndex
from e3nn import o3
import numpy as np 
import h5py 
from sklearn.metrics import r2_score 


def load_model(config_name, f_path, device):
    config = config_name.model_config
    model = build(config).to(device)
    state_dict = torch.load(f_path, map_location=device)
    model_state_dict = {}
    for key, value in state_dict.items():
        if key[:7] == 'module.':
            key = key[7:]
        model_state_dict[key] = value
    model.load_state_dict(model_state_dict)
    return model

def load_input(coord, atom_type, n_nodes, r_cutnn, device):
    """
    Function to prepare input data for the neural network model.

    Parameters:
    coord (torch.Tensor): Coordinates of atoms in the molecule. Shape: (num_atoms, 3).
    atom_type (torch.Tensor): Atomic numbers of atoms in the molecule. Shape: (num_atoms,).
    n_nodes (torch.Tensor): Number of atoms in the molecule. Shape: (1, 1).
    r_cutnn (float): Cutoff distance for constructing the neighbor list.
    device (str): Device to store the input data ('cpu' or 'cuda').

    Returns:
    input_batch (Batch): Processed input data for the neural network model.
    """
    data = {'pos': coord, 'species': atom_type, '_n_nodes': n_nodes}
    attrs = {'pos': ('node', '1x1o'), 'species': ('node','1x0e')}
    _data, _attrs = computeEdgeIndex(data, attrs, r_max=r_cutnn, r_min=0.5)
    data.update(_data)
    attrs.update(_attrs)
    input_batch = Batch(attrs, **data).to(device)
    return input_batch

atomic_num = ase.data.atomic_numbers
# we may save all the param in harmonic format 
data = np.load('../../data/resp_q/molecule/data.npz',allow_pickle=True)
coords = data['coord']; symbols = data['symbol']
labels = data['label']

species = []
for ss in symbols:
    species.append([atomic_num[s] for s in ss])

device = 'cpu'; r_cutnn = 3.5 
model = load_model(configs.config_monopole(), '../../weight/best.pt', device)
Q_samp = []; Q_label_samp = []
mol_idx = []
for idx in range(0, len(coords), 1):
    coord = torch.tensor(coords[idx], dtype=torch.float32)
    spec = torch.tensor(species[idx], dtype=torch.long)
    n_nodes = torch.ones((1,1), dtype=torch.long)*len(coord)
    input_0 = load_input(coord, spec, n_nodes, r_cutnn, device)
    q = model(input_0)['monopole']
    q0 = np.array(np.concatenate(q.tolist()))
    q1 = np.array(labels[idx])
    Q_samp.extend(np.concatenate(q.tolist()))
    Q_label_samp.extend(labels[idx])
    mol_idx.append(idx)
Q_samp = np.array(Q_samp)
Q_label_samp = np.array(Q_label_samp)
np.savez('resp_q_evaluate.npz', q_pred=Q_samp, q_label = Q_label_samp, mol_idx=mol_idx)

print(np.mean(np.abs(Q_samp-Q_label_samp)))
print(r2_score(Q_label_samp, Q_samp))

# MAE: 0.01165
# R2: 0.9896 






