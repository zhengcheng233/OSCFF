#!/usr/bin/env python
'''
基于火山镜像生成力场，支持直接基于sobtop生成力场或者基于sobtop+dihedral参数生成力场
'''
import numpy as np
import os 
import subprocess
import sys 
#from rdkit import Chem 
#import parmed 
#from deepdih.mollib.fragment import Fragmentation
#mport deepdih 
#from deepdih.utils.embedding import get_embed
#import msys 
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

def generate_gaff(f_sdf):
    # 直接基于sobtop 生成力场，仅测试参考，此处假设已有chg文件
    import msys 
    from rdkit import Chem
    mol = Chem.MolFromMolFile(f_sdf, removeHs=False)
    mol_check = msys.ConvertFromRdkit(mol)
    msys.AssignBondOrderAndFormalCharge(mol_check)
    mol_convert = msys.ConvertToRdkit(mol_check)
    Chem.MolToMolFile(mol_convert, 'mol_checked.sdf')
    # 采用sobtop生成力场
    # 需要目录下所有文件
    # 生成mol2文件 
    os.system('obabel -isdf mol_checked.sdf -omol2 -O mol.mol2')

def generate_oscff(f_sdf):
    # 直接基于sobtop 生成力场，仅测试参考，此处假设已有chg文件
    os.system('source /root/e3_layer.sh')
    os.system('source activate topo_env')
    import msys 
    from rdkit import Chem
    mol = Chem.MolFromMolFile(f_sdf, removeHs=False)
    mol_check = msys.ConvertFromRdkit(mol)
    msys.AssignBondOrderAndFormalCharge(mol_check)
    mol_convert = msys.ConvertToRdkit(mol_check)
    Chem.MolToMolFile(mol_convert, 'mol_checked.sdf')
    os.system('conda deactivate')
    os.system('source activate ffenv')
    # 采用sobtop生成力场
    # 需要目录下所有文件
    cwd = os.path.abspath(os.getcwd())
    os.system('mkdir -p sobtop')
    os.system('cp /root/sobtop/* sobtop/')
    os.system('cp mol_checked.sdf sobtop/')
    os.system('cp input.chg sobtop/')
    os.chdir('cd sobtop')
    # 生成mol2文件 
    os.system('obabel -isdf mol_checked.sdf -omol2 -O mol.mol2')
    # 运行sobtop,只生成gaff力场
    os.system('./sobtop mol.mol2 < sob_gaff.inp')
    os.chdir(f'cd {cwd}')
    os.system('conda deactivate')


#generate_gaff('mol.sdf')
# xtb input.xyz --ohess # 生成
def convert_standard_sdf(f_sdf):
    # 将sdf转换为标准格式
    # 需要需要加载的环境：source /root/e3_layer.sh; source activate topo_env; conda deactivate
    import msys 
    from rdkit import Chem
    mol = Chem.MolFromMolFile(f_sdf, removeHs=False)
     
    mol_check = msys.ConvertFromRdkit(mol)
    msys.AssignBondOrderAndFormalCharge(mol_check)
    mol_convert = msys.ConvertToRdkit(mol_check)
    Chem.MolToMolFile(mol_convert, 'mol_checked.sdf')

    mol = Chem.MolFromMolFile('mol_checked.sdf', removeHs=False)
    symbol = []
    coord = mol.GetConformer().GetPositions()
    for atom in mol.GetAtoms():
        symbol.append(atom.GetSymbol())
    with open('mol.xyz','w') as fp:
        fp.write(f'{len(symbol)}\n\n')
        for i in range(len(symbol)):
            fp.write('%s %.6f %.6f %.6f \n' % (symbol[i], coord[i][0], coord[i][1], coord[i][2]))
    
    return 

def generate_nn_charge():
    # 采用e3nn预测电荷, 环境为base source /root/e3_layer.sh
    import torch 
    import ase 
    from e3_layers.utils import build
    from e3_layers import configs
    from e3nn import o3
    import h5py 
    from e3_layers.data import Batch, computeEdgeIndex

    cwd = os.path.abspath(os.getcwd())
    atomic_num = ase.data.atomic_numbers
    device = 'cpu'; r_cutnn = 3.5
    model = load_model(configs.config_monopole(), 'best.pt', device)
    
    coord = []; symbol = []
    with open('mol.xyz','r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip().split()
            if len(line) == 4:
                symbol.append(line[0])
                coord.append([float(line[1]), float(line[2]), float(line[3])])
    species = []
    for ii in range(len(symbol)):
        species.append(atomic_num[symbol[ii]])
    coord = torch.tensor(coord, dtype=torch.float32)
    spec = torch.tensor(species, dtype=torch.long)
    n_nodes = torch.ones((1,1), dtype=torch.long)*len(coord)
    input_0 = load_input(coord, spec, n_nodes, r_cutnn, device)

    q = model(input_0)['monopole']
    q = np.array(np.concatenate(q.tolist()))
    q_net = - np.sum(q)
    q += q_net / len(q)

    assert(np.abs(np.sum(q)) < 0.0001)
    with open('nn.chg','w') as f:
        for ii in range(len(q)):
            f.write('%s %.6f %.6f %.6f %.8f \n' %(symbol[ii], coord[ii][0], coord[ii][1], coord[ii][2], q[ii]))
    return

def generate_init_oscff_1():
    # activate 环境： source /root/e3_layer.sh； source activate ffenv； conda deactivate
    cwd = os.path.abspath(os.getcwd())
    os.system('mkdir -p sobtop')
    os.system('cp /root/sobtop/* sobtop/')
    os.system('cp mol_checked.sdf sobtop/')
    os.system('cp nn.chg sobtop/')
    os.system('cd sobtop')
    os.system('obabel -isdf mol_checked.sdf -omol2 -O mol.mol2')
    os.system('xtb mol.xyz --ohess')
    # 一个只生成msem力场，一个尽可能发现miss力场
    # mol.itp & mol_miss.itp 
    os.system('./sobtop mol.mol2 < sob_oscff.inp')
    os.system('./sobtop mol.mol2 < sob_msem.inp')
    os.system(f'cd {cwd}')
    return 

def generate_init_oscff_0():
    os.system('obabel -isdf mol_checked.sdf -mol2 -O mol.mol2') 
    return 

def write_gmx(info, filename):
    with open(filename, "w") as f:
        for ii in range(len(info)):
            f.write(f'[ {info[ii]["key"]} ]\n')
            for line in info[ii]["data"]:
                f.write(line+"\n")
            f.write("\n")

def read_topo(topo):
    with open(topo, 'r') as f:
        info = []; key = None
        for line in f:
            if len(line.strip()) == 0:
                continue
            if line.startswith(';'):
                continue
            elif line.startswith('['):
                key = line.strip()
                info.append({})
                info[-1]["key"] = key.strip()[1:-1].strip().lower()
                info[-1]["data"] = []
                continue
            info[-1]["data"].append(line.strip())
    return info


def generate_oscff():
    # 基于deepdih 寻找不被支持的原子，然后采用oscff parameter 补充相应的参数
    # source activate ffenv 
    import deepdih 
    import pickle
    from deepdih.utils.embedding import get_embed
    from rdkit import Chem 
    
    with open('rot_torsion_params.pkl','rb') as fp:
        dih_parm = pickle.load(fp)
    embeds = dih_parm.embeddings
    params = dih_parm.parameters
    # 根据mol.top & mol.itp + dih_parm 更新力场
    mol = Chem.SDMolSupplier('mol_checked.sdf', sanitize=True, removeHs=False)[0]
    embed = get_embed(mol, layers=1)
    # 读取itp文件
    params_ori = read_topo('mol_miss.itp'); params_mSem = read_topo('mol_msem.itp')
    # 更新结果
    for ii in range(len(params_ori)):
        if params_ori[ii]['key'] == 'bonds':
            bond_data = params_ori[ii]["data"]
            for idx, line in enumerate(bond_data):
                line = line.strip().split()
                if 'missing' in line:
                    params_ori[ii]['data'][idx] = params_mSem[ii]['data'][idx]
        elif params_ori[ii]['key'] == 'angles':
            angle_data = params_ori[ii]["data"]
            for idx, line in enumerate(angle_data):
                line = line.strip().split()
                if 'missing' in line:
                    params_ori[ii]['data'][idx] = params_mSem[ii]['data'][idx]
        elif 'dihedrals' in params_ori[ii]['key'] and 'improper' not in params_ori[ii]['key']:
            tor_data = params_ori[ii]["data"]
            cleaned_data_tor = {}
            for line in tor_data:
                line = line.strip().split()
                if 'missing' not in line:
                    key = tuple([int(line[0])-1, int(line[1])-1, int(line[2])-1, int(line[3])-1])
                    assert(int(line[4]) == 9)
                    val = [int(line[4]), float(line[5]), float(line[6]), int(line[7])]
                    if key[1] > key[2]:
                        key = tuple([key[3], key[2], key[1], key[0]])
                    if key not in cleaned_data_tor:
                        cleaned_data_tor[key] = []
                    cleaned_data_tor[key].append(val)
            for idx, line in enumerate(tor_data):
                line = line.strip().split()
                if 'missing' in line:
                    key = [int(line[0])-1, int(line[1])-1, int(line[2])-1, int(line[3])-1]
                    if key[1] > key[2]:
                        key = [key[3], key[2], key[1], key[0]]
                    torsion_embed = (embed[key,:] + embed[key[::-1],:])/2
                    torsion_embed = torsion_embed.ravel()
                    dis = np.linalg.norm(torsion_embed - embeds, axis=1)
                    if np.min(dis) < 0.01:
                        idx = np.argmin(dis)
                        if tuple(key) not in cleaned_data_tor:
                            cleaned_data_tor[tuple(key)] = []
                        val = params[idx]
                        added_lines = []
                        for order in range(6):
                            prm_val = val[order].item()
                            if abs(prm_val) < 1e-4:
                                continue
                            if prm_val > 0.0:
                                added_lines.append((9, 0.0, prm_val, order+1))
                            else:
                                added_lines.append((9, 180.0, -prm_val, order+1))
                        if len(added_lines) > 0:
                            cleaned_data_tor[tuple(key)] = added_lines
                    else:
                        line0 = params_mSem[ii]['data'][idx].strip().split()
                        added_lines = [(int(line0[4]), float(line0[5]), float(line0[6]))]
                        cleaned_data_tor[tuple(key)] = added_lines
            tor_text = []
            for key in cleaned_data_tor:
                val = cleaned_data_tor[key]
                for vv in val:
                    if vv[0] == 9:
                        tor_text.append(f'{key[0]+1:>5} {key[1]+1:>5} {key[2]+1:>5} {key[3]+1:>5} {vv[0]:>5} {vv[1]:6.2f} {vv[2]:>16.8f} {vv[3]:>5}')
                    else:
                        tor_text.append(f'{key[0]+1:>5} {key[1]+1:>5} {key[2]+1:>5} {key[3]+1:>5} {vv[0]:>5} {vv[1]:6.2f} {vv[2]:>16.8f}')
            params_ori[ii]['data'] = tor_text
    tor_mSem = [] 
    for term in range(len(params_ori)):
        if 'dihedrals' in params_ori[term]['key'] and 'improper' not in params_ori[term]['key']:
            tor_data = params_ori[term]["data"]
            for line in tor_data:
                line = line.strip().split()
                if int(line[4]) == 2:
                    tor_mSem.append((int(line[0])-1, int(line[3])-1))
    pairs_line = []
    for term in range(len(params_ori)):
        if 'pairs' in params_ori[term]['key']:
            pairs_data = params_ori[term]['data']
            for line in pairs_data:
                line_0 = line.strip().split()
                key_0 = (int(line_0[0])-1, int(line_0[1])-1)
                key_1 = (int(line_0[1])-1, int(line_0[0])-1)
                if key_0 in tor_mSem or key_1 in tor_mSem:
                    continue
                pairs_line.append(line)
    for term in range(len(params_ori)):
        if 'pairs' in params_ori[term]['key']:
            params_ori[term]['data'] = pairs_line
    write_gmx(params_ori, 'mol.itp')
    #os.system('mv mol.itp ../')
    return 

if __name__ == '__main__':
    opera = sys.argv[1]
    if opera == 'convert_sdf':
        convert_standard_sdf('mol.sdf')
    elif opera == 'calc_q':
        generate_nn_charge()
    elif opera == 'gen_itp':
        generate_oscff()
    elif opera == 'gen_gaff':
        generate_gaff('mol.sdf')

    


    

