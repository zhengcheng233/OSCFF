#!/usr/bin/env python 
import os
import numpy as np  
from glob import glob 
import pickle 
from rdkit import Chem
import parmed
from deepdih.mollib.fragment import Fragmentation
import sys
from deepdih.utils.geometry import calc_max_disp, calc_rmsd
import deepdih
from multiprocessing import Pool
from deepdih.utils.embedding import get_embed
import torch
import sys 
import math 

opera = sys.argv[1]

def read_topo(f_itp):
    with open(f_itp, 'r') as f:
        lines = f.readlines()
        info_0 = []; key = None 
        for line in lines:
            if len(line.strip()) == 0:
                continue
            if line.startswith(';'):
                continue
            elif line.startswith('['):
                key = line.strip()
                info_0.append({})
                info_0[-1]["key"] = key.strip()[1:-1].strip().lower()
                info_0[-1]["data"] = []
                continue
            info_0[-1]["data"].append(line.strip())
    return info_0 

def write_gmx(info, filename):
    with open(filename, "w") as f:
        for ii in range(len(info)):
            f.write(f'[ {info[ii]["key"]} ]\n')
            for line in info[ii]["data"]:
                f.write(line+"\n")
            f.write("\n")

if opera == 'complete_parm':
    # 补齐力场参数, 针对分子片段补齐，实际后续针对分子也需要补齐
    # 二面角参数数据库
    with open('../../weight/rot_torsion_params.pkl', 'rb') as f:
        params = pickle.load(f)
    torsion_embeds = params.embeddings
    torsion_params = params.parameters 

    # 读取分子
    #f_names = glob('../../data/dihedral_scan/mol_*/mol.itp')
    f_names = [f'../../data/dihedral_single/case_{sys.argv[2]}/mol.itp']
    
    #f_names = ['./semisucced/mol_178/mol.itp']

    for f_name in f_names:
        f_dir = os.path.dirname(f_name)
        f_sdf = os.path.join(f_dir, 'mol.sdf')
        f_itp = os.path.join(f_dir, 'mol.itp')
        mol = deepdih.utils.read_sdf(f_sdf)
        embed = get_embed(mol[0], layers=1)
        
        info_0 = read_topo(f_itp)
        # 需要更新参数为miss的二面角， 参考下deepdih的写法
        # 找到miss的dihedral 
        miss_torsion = []
        for info in info_0:
            if 'dihedrals' in info['key'] and 'improper' not in info['key']:
                dihedral_data = info['data']
                for idx, line in enumerate(dihedral_data):
                    line = line.strip().split()
                    if 'missing' in line:
                        dihedral = [int(line[0])-1, int(line[1])-1, int(line[2])-1, int(line[3])-1]
                        miss_torsion.append(dihedral)
        # 找到除miss的dihedral参数以外的二面角参数
        for info in info_0:
            if 'dihedrals' in info['key'] and 'improper' not in info['key']:
                tor_data = info['data']
                cleaned_data = {}
                for line in tor_data:
                    line = line.strip().split()
                    if 'missing' not in line:
                        key = tuple([int(line[0])-1, int(line[1])-1, int(line[2])-1, int(line[3])-1])
                        if int(line[4]) == 2:
                            val = [int(line[4]), float(line[5]), float(line[6])]
                        elif int(line[4]) == 9:
                            val = [int(line[4]), float(line[5]), float(line[6]), int(line[7])]
                        if key[1] > key[2]:
                            key = (key[3], key[2], key[1], key[0])
                        if key not in cleaned_data:
                            cleaned_data[key] = []
                        cleaned_data[key].append(val)
        # 补上miss的二面角参数
        for torsion in miss_torsion:
            torsion_embed = (embed[torsion,:] + embed[torsion[::-1],:]) / 2. 
            torsion_embed = torsion_embed.ravel()
            dis = np.linalg.norm(np.array(torsion_embeds) - torsion_embed, axis=1)
            if np.min(dis) < 1e-2:
                idx = np.argmin(dis)
                tor_parm = torsion_params[idx]
            else:
                print('error in dihedral') 
                print(f_name)
            ii, jj, kk, ll = torsion
            if jj > kk:
                ii, jj, kk, ll = ll, kk, jj, ii
            added_lines = []
            for order in range(6):
                prm_val = tor_parm[order].item()
                if abs(prm_val) < 1e-4:
                    continue

                if prm_val > 0.0:
                    added_lines.append((9, 0.0, prm_val, order+1))
                else:
                    added_lines.append((9, 180.0, -prm_val, order+1))
            if len(added_lines) > 0:
                cleaned_data[(ii, jj, kk, ll)] = added_lines
        
        tor_text = [] 
        for key in cleaned_data:
            for val in cleaned_data[key]:
                if val[0] == 9:
                    tor_text.append(f'{key[0]+1:>5} {key[1]+1:>5} {key[2]+1:>5} {key[3]+1:>5} {val[0]:>5} {val[1]:6.2f} {val[2]:>16.8f} {val[3]:5}')
                else:
                    tor_text.append(f'{key[0]+1:>5} {key[1]+1:>5} {key[2]+1:>5} {key[3]+1:>5} {val[0]:>5} {val[1]:6.2f} {val[2]:>16.8f}')

        for term in range(len(info_0)):
            if 'dihedrals' in info_0[term]['key'] and 'improper' not in info_0[term]['key']:
                info_0[term]['data'] = tor_text
        try:
            os.mkdir(os.path.join(f_dir, 'mol_complete'))
        except:
            print('exist')
        f_itp = os.path.join(f_dir, 'mol_complete', 'mol.itp')
        write_gmx(info_0, f_itp)
        f_top = os.path.join(f_dir,'mol.top')
        os.system(f'cp {f_top} {os.path.join(f_dir, "mol_complete", "mol.top")}')

elif opera == 'evaluate_ff':
    # 统计更新后力场的误差，与qm结果比较
    #f_names = glob('../../data/dihedral_scan/mol_*/dih_scan.npz')
    f_names = glob('../../data/dihedral_scan/mol_*/train_data.npz')
    info = []; tot_mse = []; tot_num = []
    qm_energy = []; mm_energy = []; labels = []

    for f_name in f_names:
        try:

            f_dir = os.path.dirname(f_name)
            f_top = os.path.join(f_dir, 'mol_complete', 'mol.top')
            f_sdf = os.path.join(f_dir, 'mol.sdf')
            f_scan = os.path.join(f_dir, 'mm_scan_remain.sdf')
            
            mol = deepdih.utils.read_sdf(f_sdf)[0]

            with open(f_scan, 'r') as f:
                lines = f.readlines()
                if len(lines) < 1:
                    continue 
            data = np.load(f_name,allow_pickle=True)
            #coords = data['coord_mmopt']
            E_qm = data['qm_from_mmopt']
            calculator = deepdih.calculators.GromacsTopCalculator(mol, f_top)
            mols_scan = Chem.SDMolSupplier(f_scan, sanitize=True, removeHs=False)
            mm_conformers = [deepdih.geomopt.recalc_energy(c, calculator) for c in mols_scan]
            E_mm = [float(c.GetProp('ENERGY')) for c in mm_conformers]
            E_qm = np.array(E_qm); E_mm = np.array(E_mm)
            E_qm = E_qm - np.mean(E_qm); E_mm = E_mm - np.mean(E_mm)
            lab = any(math.isnan(x) for x in E_mm)
            if lab:
                print(f_name)
            else:
                qm_energy.extend(E_qm); mm_energy.extend(E_mm)
                rmse_e = np.sqrt(np.mean((E_qm - E_mm)**2))
                info.append({'rmse_e': rmse_e, 'mol': f_dir})
                tot_mse.append(rmse_e**2)
                tot_num.append(len(E_qm))
                #labels.append(f_dir)
        except:
            print(f_name)


    with open('static.txt','w') as fp:
        for ii in info:
            fp.write(f'{ii["mol"]}: {ii["rmse_e"]}\n')
    tot_mse = np.array(tot_mse, dtype=object); tot_num = np.array(tot_num, dtype=object)
    qm_energy = np.array(qm_energy, dtype=object); mm_energy = np.array(mm_energy, dtype=object)
    np.savez('check.npz', tot_mse=tot_mse, tot_num=tot_num, \
                          qm_energy=qm_energy, mm_energy=mm_energy)
    
elif opera == 'evaluate_ff_single':
    # 统计更新后力场的误差，与qm结果比较
    #f_names = glob('./semisucced/mol_*/train_data.npz') + glob('./succed/mol_*/train_data.npz') + \
    #          glob('./tryagain/mol_*/mol.itp')
    
    # mol_223; mol_336; mol_22
    f_name = f'../../data/dihedral_single/case_{sys.argv[2]}/train_data.npz'
    info = []; tot_mse = []; tot_num = []
    qm_energy = []; mm_energy = []; labels = []

    #for f_name in f_names:
    #    try:
    f_dir = os.path.dirname(f_name)
    f_top = os.path.join(f_dir, 'mol_complete', 'mol.top')
    f_sdf = os.path.join(f_dir, 'mol.sdf')
    f_scan = os.path.join(f_dir, 'mm_scan_remain.sdf')
    f_torsion = os.path.join(f_dir, 'torsion_key.npy')
    mol = deepdih.utils.read_sdf(f_sdf)[0]
    
    with open(f_scan, 'r') as f:
        lines = f.readlines()
    
    data = np.load(f_name,allow_pickle=True)
    #coords = data['coord_mmopt']
    E_qm = data['qm_from_mmopt']
    calculator = deepdih.calculators.GromacsTopCalculator(mol, f_top)
    mols_scan = Chem.SDMolSupplier(f_scan, sanitize=True, removeHs=False)
    mm_conformers = [deepdih.geomopt.recalc_energy(c, calculator) for c in mols_scan]
    E_mm = [float(c.GetProp('ENERGY')) for c in mm_conformers]
    E_qm = np.array(E_qm); E_mm = np.array(E_mm)
    qm_energy.append(E_qm); mm_energy.append(E_mm)
    E_qm = E_qm - np.mean(E_qm); E_mm = E_mm - np.mean(E_mm)
    rmse_e = np.sqrt(np.mean((E_qm - E_mm)**2))
    info.append({'rmse_e': rmse_e, 'mol': f_dir})
    tot_mse.append(rmse_e**2)
    tot_num.append(len(E_qm))
    coord_mmopt = data['coord_mmopt']
    reason = data['reason']
    torsion_key = np.load(f_torsion)
    np.savez(f'{f_dir}/check_sing.npz', rmse_e=rmse_e, E_qm=E_qm, E_mm=E_mm, \
                coord=coord_mmopt, torsion=torsion_key, reason=reason)
    #labels.append(f_dir)
#except:
#    print(f_name)











