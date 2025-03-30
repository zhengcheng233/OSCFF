#!/usr/bin/env python 
import numpy as np 
from rdkit import Chem 
from ase.data import atomic_numbers, atomic_masses
import os 
import sys 
from multiprocessing import Pool

def generate_slurm(f_comands, idx, nproc, task_name):
    f_slurm = f'./{task_name}_{idx}.slurm'
    with open(f_slurm, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#SBATCH -p kshcnormal\n')
        f.write(f'#SBATCH -J {task_name}_{idx}\n')
        f.write('#SBATCH -N 1\n')
        f.write(f'#SBATCH -n {nproc}\n')
        for comand in f_comands:
            f.write(f'{comand}\n')
    return

def generate_slurm_gpu(f_comands, idx, nproc, task_name):
    f_slurm = f'./{task_name}_{idx}.slurm'
    with open(f_slurm, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#SBATCH -p kshdnormal\n')
        f.write(f'#SBATCH -J {task_name}_{idx}\n')
        f.write('#SBATCH -N 1\n')
        f.write(f'#SBATCH --ntasks-per-node=1\n')
        f.write(f'#SBATCH --cpus-per-task={nproc}\n')
        f.write(f'#SBATCH --gres=dcu:1\n')
        for comand in f_comands:
            f.write(f'{comand}\n')
    return

opera = sys.argv[1] 

if opera == 'test_box':
    mol_0 = Chem.SDMolSupplier('mol_checked.sdf',removeHs=False)[0]
    symbol = []
    for atom in mol_0.GetAtoms():
        symbol.append(atom.GetSymbol())
    tot_mass = 0.
    for ss in symbol:
        tot_mass += atomic_masses[atomic_numbers[ss]]
    tot_mass = tot_mass / (6.02214076e23) * 100
    # 简单认为半导体薄膜的密度为1.3g/cm^3
    volum = tot_mass / 1.2
    box_length = volum**(1./3) * (10 ** 8) * 2.5 # angstrom
    #box_length = 250 # angstrom
    # 生成packmol命令参数
    with open('packmol.inp', 'w') as f:
        f.write('tolerance 2.5\n')
        f.write('add_box_sides 2.5\n')
        f.write('output mol_box.pdb \n')
        f.write('structure mol.pdb\n')
        f.write('  number 100\n')
        f.write(f'  inside cube 0. 0. 0. {box_length} \n')
        f.write('end structure\n')
    # 运行packmol生成盒子的pdb 
    os.system('obabel -i sdf mol_checked.sdf -O mol.pdb')
    os.system('packmol < packmol.inp')

    f_path = '/public/home/chengz/photomat/md_eg_for_paper'
    os.system(f'cp {f_path}/mol_gaff.top .')
    os.system(f'cp {f_path}/mol_oscff.top .')
    # 将pdb转换为gro文件
    #os.system('module purge && source /public/home/chengz/apprepo/gromacs_threadMPI/2023.2-dtk23.10_hpcx241/scripts/env.sh && gmx editconf -f mol_box.pdb -o mol.gro')
    # 修改top文件
    #if os.path.exists('mol.top.bak') is False:
    #    os.system('cp mol.top mol.top.bak')
    
    #with open('mol.top.bak', 'r') as f:
    #    lines = f.readlines()
    
    #lines1 = []
    #for line in lines:
    #    line1 = line.strip().split()
    #    if len(line1) == 2 and line1[0] == 'mol' and line1[1] == '1':
    #        line = f'mol      100\n'
    #    lines1.append(line)
    #with open('mol.top', 'w') as f:
    #    for line in lines1:
    #        f.write(line)

elif opera == 'run_md':
    f_dir = './mdtest_foroscff/mol_300/final' 
    # 第一次min 
    # gmx editconf -f mol_box.pdb -o mol.gro
    # gmx grompp -f min.mdp -p mol.top -c mol.gro -o mol.tpr
    # gmx mdrun -v -deffnm mol 一般默认该参数，可自己修改
    # 轨迹转换文件 gmx trjconv -s mol.tpr -f mol.trr -o output0.pdb -pbc whole -center
    # gmx trjconv -s mol.tpr -f mol.trr -o mol2.gro -pbc mol -ur compact -center
    # 两次min; 然后NPT md模拟， 最简单的观察盒子体积是否平衡 
    # 第二次min, 取消水分子的约束
    # gmx grompp -f min2.mdp -o mol2.tpr -p mol.top -po mol2.mdp -c mol2.gro -t mol2.trr  
    # gmx mdrun -v -deffnm mol2
    # gmx trjconv -s mol2.tpr -f mol2.trr -o mol2.pdb -pbc whole -center
    # NVT 平衡
    # gmx trjconv -s mol2.tpr -f mol2.trr -o mol3.gro -pbc mol -ur compact -center
    # gmx grompp -f md.mdp -o mol3.tpr -p mol.top -c mol3.gro -t mol2.trr 
    # gmx mdrun -deffnm mol3
    # gmx trjconv -s mol3.tpr -f mol3.trr -o mol3.pdb -pbc whole -center
    # NPT 平衡
    # gmx trjconv -s mol3.tpr -f mol3.trr -o mol4.gro -pbc mol -ur compact -center
    # gmx grompp -f md_npt.mdp -o mol4.tpr -p mol.top -c mol4.gro -t mol3.trr
    # gmx mdrun -deffnm mol4
    # 

elif opera == 'gen_box_inbatch':
    # we have two npz files, mol_for_md.npz & mrtadf_osc.npz 
    npz_dir = './forcefield'
    data = np.load(f'{npz_dir}/mol_for_md.npz',allow_pickle=True)
    #npz_dir = './forcefield_mrtadf'
    #data = np.load(f'{npz_dir}/mrtadf_osc.npz',allow_pickle=True)
    mol_num = len(data['smi'])
    mol_dirs = []
    for ii in range(mol_num):
        top_dir = os.path.join(f'{npz_dir}/mol_{ii}','final')
        if os.path.exists(os.path.join(top_dir,'mol.itp')):
            f_path = os.path.join(f'{npz_dir}/mol_{ii}','final','mol.gro')
            if os.path.exists(f_path):
                continue
            mol_dirs.append(top_dir)

    # 需要更新top文件--> 100 分子
    # 需要packmol生成pdb--> gro文件
    task_set = []
    for ii in range(500):
        task_set.append([])
    for ii, f_dir in enumerate(mol_dirs):
        task_set[ii%500].append(f_dir)
    cwd = os.path.abspath(os.getcwd())
    for ii in range(500):
        commands = []; task_single = task_set[ii]
        commands.append('module purge')
        commands.append('source /public/home/chengz/apprepo/gromacs/2023-gcc930_intelmpi17/scripts/env.sh')

        for jj in task_single:
            commands.append(f'cd {jj}')
            commands.append(f'python {cwd}/gen_sol_box.py test_box')
            commands.append(f'gmx_mpi editconf -f mol_box.pdb -o mol.gro')
            commands.append(f'rm mol_box.pdb')
            commands.append(f'cd {cwd}')
        generate_slurm(commands, ii, 1, 'fp')

elif opera == 'clean_data':
    npz_dir = './forcefield'
    data = np.load(f'{npz_dir}/mol_for_md.npz',allow_pickle=True)
    #npz_dir = './forcefield_mrtadf'
    #data = np.load(f'{npz_dir}/mrtadf_osc.npz',allow_pickle=True)
    mol_num = len(data['smi'])
    mol_dirs = []
    for ii in range(mol_num):
        top_dir = os.path.join(f'{npz_dir}/mol_{ii}','final')
        mol_dirs.append(top_dir)

    task_set = []
    for ii in range(500):
        task_set.append([])
    for ii, f_dir in enumerate(mol_dirs):
        task_set[ii%500].append(f_dir)
    for ii in range(500):
        commands = []; task_single = task_set[ii]
        for jj in task_single:
            commands.append(f'rm {jj}/mol_box.pdb')
        generate_slurm(commands, ii, 1, 'clean')

elif opera == 'runmd_inbatch_0':
    # 第一次md模拟，快速压缩盒子，md_0.mdp, 
    # 此外我们还会检查min任务是否成功（力场是否合理，每次md都会检查） 
    # we have two npz files, mol_for_md.npz & mrtadf_osc.npz 
    npz_dir = './forcefield'
    data = np.load(f'{npz_dir}/mol_for_md.npz',allow_pickle=True)
    # 控制文件位置
    f_min_path = '/public/home/chengz/photomat/min.mdp'
    f_md_path = '/public/home/chengz/photomat/md_0.mdp'
    py_path = '/public/home/chengz/photomat/check_min.py'

    #npz_dir = './forcefield_mrtadf'
    #data = np.load(f'{npz_dir}/mrtadf_osc.npz',allow_pickle=True)
    def check_mol_dir(ii):
        ii = str(ii)
        top_dir = os.path.join(f'{npz_dir}/mol_{ii}', 'final')
        energy_file_path = os.path.join(top_dir, 'energy.xvg')
        q_file_path = os.path.join(f'{npz_dir}/mol_{ii}', f'input.chg')
        if os.path.exists(energy_file_path):
            with open(energy_file_path, 'r') as f:
                lines = f.readlines()
                line = lines[0].strip().split()
                
                if line[0] == 'fail':
                    return None 
                else:
                    #mol_dirs.append(top_dir)
                    if len(lines) > 10:
                        return None
                    else:
                        large_q = False
                        with open(q_file_path, 'r') as f:
                            lines = f.readlines()
                            for line in lines:
                                line = line.strip().split()
                                if len(line) == 5:
                                    if float(line[4]) > 1.8:
                                        large_q = True
                                        break
                        if large_q:
                            return None
                        else:
                            return top_dir
        else:
            print(f'Not exist: {top_dir}')
            return None


    mol_num = len(data['smi'])
    #mol_num = 2000 
    #results = []
    with Pool(10) as pool:
        results = pool.map(check_mol_dir, [i for i in range(mol_num)])
    #print(results)
    mol_dirs = [ii for ii in results if ii is not None]
    #print(mol_dirs)


    # 考虑到存储问题，我们只保存md模拟最后的gro文件，此外，我们还需采用gmx energy 分析数据
    # 用于判断最终的gro文件是平衡结构
    task_set = []
    for ii in range(2000):
        task_set.append([])
    for ii, f_dir in enumerate(mol_dirs):
        task_set[ii%2000].append(f_dir)
    cwd = os.path.abspath(os.getcwd())

    for ii in range(2000):
        commands = []
        commands.append('module purge')
        commands.append('source /public/home/chengz/apprepo/gromacs_threadMPI/2023.2-dtk23.10_hpcx241/scripts/env.sh')
        for jj in task_set[ii]:
            mol_name = os.path.basename(os.path.dirname(jj))
            local_path = os.path.abspath(jj)
            tmp_path = f'/tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}'
            commands.append(f'mkdir -p /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {local_path}/min.gro /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {f_min_path} /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {f_md_path} /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {local_path}/mol.top /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {local_path}/mol.itp /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {py_path} /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {local_path}/mol_checked.sdf /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cd /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            # 将min结构导出
            # md模拟过程，我们发现min还是在cpu跑快，因为gpu用不上
            # md可能分两步，压缩盒子&平衡结构, 若密度稳定，压缩系数变小，仍很小就维持压缩系数压
            #commands.append(f'python check_min.py')
            commands.append(f'gmx grompp -f md_0.mdp -p mol.top -c min.gro -o md.tpr > 0.out 2>&1')   
            commands.append('sleep 1')         
            commands.append(f'gmx mdrun -gpu_id 0 -ntmpi 1 -ntomp 7 -nb gpu -pme gpu -deffnm md > tmp.out 2>&1')
            commands.append('sleep 1')
            commands.append(f'echo -e "Density\\n" | gmx energy -f md.edr -o energy.xvg > 1.out 2>&1')
            commands.append('sleep 1')
            commands.append(f'echo -e "0\\n0\\n" | gmx trjconv -f md.xtc -s md.tpr -dump 999999 -o lst.gro -pbc mol -center > 2.out 2>&1')
            commands.append('sleep 1')           
            #commands.append(f'gmx editconf -f lst.gro -o lst.xyz')
            commands.append(f'python check_min.py')
            commands.append(f'cp {tmp_path}/energy.xvg {local_path}')
            commands.append(f'cp {tmp_path}/md.gro {local_path}/min.gro')
            commands.append(f'cp {tmp_path}/res.xvg {local_path}')
            commands.append(f'rm {local_path}/mol.gro')
            commands.append(f'cd {cwd}')
            commands.append(f'rm -rf /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append('sleep 1')
        generate_slurm_gpu(commands, ii, 8, 'md')

elif opera == 'runmd_inbatch_1':
    # 第二次md模拟，平衡盒子，md_1.mdp, md_1.mdp已经设置了continuation=yes, 可以将上一轮的md.gro 转为min.gro，继续模拟
    # 此外我们还会检查min任务是否成功（力场是否合理，每次md都会检查） 
    # we have two npz files, mol_for_md.npz & mrtadf_osc.npz 
    npz_dir = './forcefield'
    data = np.load(f'{npz_dir}/mol_for_md.npz',allow_pickle=True)
    # 控制文件位置
    f_min_path = '/public/home/chengz/photomat/min.mdp'
    # md_path0用于密度已经平衡的盒子密度大于850 kg/m^3
    # md_path1用于未平衡的盒子，密度小于850 kg/m^3
    # 这次我们需要把lst.gro保存，同时删除mol.gro, 保存min.gro (md.grocp来的min.gro)
    f_md_path0 = '/public/home/chengz/photomat/md_1.mdp'
    f_md_path1 = '/public/home/chengz/photomat/md_2.mdp'
    f_md_path2 = '/public/home/chengz/photomat/md_3.mdp'
    py_path = '/public/home/chengz/photomat/check_min.py'
    
    #npz_dir = './forcefield_mrtadf'
    #data = np.load(f'{npz_dir}/mrtadf_osc.npz',allow_pickle=True)
    mol_num = len(data['smi'])
    mol_dirs = []; md_commands = []
    def process_mol_dir(ii):
        top_dir = os.path.join(f'{npz_dir}/mol_{ii}','final')
        q_file_path = os.path.join(f'{npz_dir}/mol_{ii}', f'input.chg')
        if os.path.exists(os.path.join(top_dir,'energy.xvg')):
            with open(os.path.join(top_dir,'energy.xvg'),'r') as f:
                lines = f.readlines(); line_final = lines[-1].strip().split()
                if len(lines) > 30:
                    line_final0 = lines[-2].strip().split()
                    line_final1 = lines[-3].strip().split()
                    # 之后还是按850 kg/m^3来判断
                    if float(line_final[-1]) < 800 and float(line_final0[-1]) < 800 and float(line_final1[-1]) < 800:
                        f_md_path = f_md_path1
                    elif float(line_final[-1]) < 950:
                        f_md_path = f_md_path2
                    else:
                        f_md_path = f_md_path0
                    if float(line_final[-1]) < 100 and float(line_final0[-1]) < 100 and float(line_final1[-1]) < 100:
                        f_md_path = None
                    if os.path.exists(os.path.join(top_dir, 'energy_1.xvg')):
                        f_md_path = None 
                else:
                    f_md_path = None 
            large_q = False
            if os.path.exists(q_file_path):
                with open(q_file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip().split()
                        if len(line) == 5:
                            if float(line[4]) > 1.8:
                                large_q = True
                                break
            else:
                large_q = True
            if large_q:
                return None, None
            if os.path.exists(os.path.join(top_dir,'res.xvg')):
                with open(os.path.join(top_dir,'res.xvg'),'r') as f:
                    lines = f.readlines()
                if 'fail' in lines[0] or f_md_path is None:
                    return None, None
                else:
                    return top_dir, f_md_path
            else:
                return None, None
            
        else:
            print(f'Not exist: {top_dir}')
            return None, None
    
    # 考虑到存储问题，我们只保存md模拟最后的gro文件，此外，我们还需采用gmx energy 分析数据
    # 用于判断最终的gro文件是平衡结构
    with Pool(10) as pool:
        results = pool.map(process_mol_dir, [i for i in range(mol_num)])
    mol_dirs = []; md_commands = []
    for ii in results:
        if ii[0] is not None:
            mol_dirs.append(ii[0])
            md_commands.append(ii[1])

    task_set = []; md_command_sets = []
    for ii in range(2000):
        task_set.append([])
        md_command_sets.append([])
    for ii, f_dir in enumerate(mol_dirs):
        task_set[ii%2000].append(f_dir)
        md_command_sets[ii%2000].append(md_commands[ii])
    
    cwd = os.path.abspath(os.getcwd())
    for ii in range(2000):
        commands = []
        commands.append('module purge')
        commands.append('source /public/home/chengz/apprepo/gromacs_threadMPI/2023.2-dtk23.10_hpcx241/scripts/env.sh')
        for idx, jj in enumerate(task_set[ii]):
            mol_name = os.path.basename(os.path.dirname(jj))
            local_path = os.path.abspath(jj)
            tmp_path = f'/tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}'
            f_md_path_final = md_command_sets[ii][idx]
            commands.append(f'mkdir -p /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {local_path}/min.gro /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {f_min_path} /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {f_md_path_final} /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}/md.mdp')
            commands.append(f'cp {local_path}/mol.top /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {local_path}/mol.itp /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {py_path} /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {local_path}/mol_checked.sdf /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cd /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'gmx grompp -f md.mdp -p mol.top -c min.gro -o md.tpr > 0.out 2>&1')            
            # 可以将输出保存>1.out然后check 
            commands.append('sleep 1')
            commands.append(f'gmx mdrun -gpu_id 0 -ntmpi 1 -ntomp 7 -nb gpu -pme gpu -deffnm md > tmp.out 2>&1')
            commands.append('sleep 1')
            commands.append(f'echo -e "Density\\n" | gmx energy -f md.edr -o energy.xvg > 1.out 2>&1')
            commands.append('sleep 1')
            commands.append(f'echo -e "0\\n0\\n" | gmx trjconv -f md.xtc -s md.tpr -dump 999999 -o lst.gro -pbc mol -center > 2.out 2>&1')
            #commands.append(f'gmx editconf -f lst.gro -o lst.xyz')
            commands.append('sleep 1')
            commands.append(f'python check_min.py')
            # 可能需要检查energy_1.xvg, 如果密度变化过于剧烈。。。第三轮md模拟平衡，还是很小直接放弃
            commands.append(f'cp {tmp_path}/energy.xvg {local_path}/energy_1.xvg')
            commands.append(f'cp {tmp_path}/lst.gro {local_path}')
            commands.append(f'cp {tmp_path}/res.xvg {local_path}')
            commands.append(f'cp {tmp_path}/md.gro {local_path}/min.gro')
            commands.append(f'rm {local_path}/mol.gro')
            commands.append(f'cd {cwd}')
            commands.append(f'rm -rf /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append('sleep 1')
        generate_slurm_gpu(commands, ii, 8, 'md')

elif opera == 'runmd_inbatch_2':
    #第三轮，如果density中最大值超过了900，且密度不稳定，我们就用低频率压缩得到最后的值
    # 如果第二轮最后一个密度远大于第一轮最后一个密度，且第二轮密度<800, 我们用高频率再压缩
    # 合理标准，密度大于900，且密度稳定（方差 取倒数前十个值不怎么变）
    npz_dir = './forcefield'
    data = np.load(f'{npz_dir}/mol_for_md.npz',allow_pickle=True)
    # 控制文件位置
    f_min_path = '/public/home/chengz/photomat/min.mdp'
    # md_path0用于密度已经平衡的盒子密度大于850 kg/m^3
    # md_path1用于未平衡的盒子，密度小于850 kg/m^3
    # 这次我们需要把lst.gro保存，同时删除mol.gro, 保存min.gro (md.grocp来的min.gro)
    # 用于保证密度稳定的
    f_md_path0 = '/public/home/chengz/photomat/md_1.mdp'
    # 用于快速加压的
    f_md_path1 = '/public/home/chengz/photomat/md_2.mdp'
    # 快速加压与密度稳定之间的
    f_md_path2 = '/public/home/chengz/photomat/md_3.mdp'
    f_md_path3 = '/public/home/chengz/photomat/md_4.mdp'
    # 检查是否合理的
    py_path = '/public/home/chengz/photomat/check_min.py'
    
    #npz_dir = './forcefield_mrtadf'
    #data = np.load(f'{npz_dir}/mrtadf_osc.npz',allow_pickle=True)
    mol_num = len(data['smi'])
    mol_dirs = []; md_commands = []
    def process_mol_dir(ii):
        top_dir = os.path.join(f'{npz_dir}/mol_{ii}','final')
        energy_0_path = os.path.join(top_dir, 'energy.xvg')
        energy_1_path = os.path.join(top_dir, 'energy_1.xvg')
        do_md = True 
        if os.path.exists(energy_0_path) and os.path.exists(energy_1_path):
            density_1 = []; density_0 = []; time1 = 0. 
            with open(energy_0_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 30:
                    for line in lines[-10:]:
                        line = line.strip().split()
                        if len(line) == 2:
                            density_0.append(float(line[-1]))
            with open(energy_1_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 30:
                    for line in lines[-10:]:
                        line = line.strip().split()
                        if len(line) == 2:
                            density_1.append(float(line[-1]))
                            time1 = float(line[0])
            density_0 = np.array(density_0); density_1 = np.array(density_1)
            var_0 = np.sqrt(np.var(density_0)); var_1 = np.sqrt(np.var(density_1))
            if time1 < 1500:
                do_md = False
            if np.mean(density_1) > 900 and var_1 / np.mean(density_1) < 0.01:
                do_md = False
            if np.mean(density_1) < 400:
                do_md = False 
            if do_md == True:
                if np.mean(density_1) < 800:
                    f_md_path = f_md_path1
                if np.mean(density_1) > 800 and np.mean(density_1) < 900:
                    f_md_path = f_md_path2
                if np.mean(density_1) > 900 and var_1 / np.mean(density_1) < 0.03:
                    f_md_path = f_md_path0
                if np.mean(density_1) > 900 and var_1 / np.mean(density_1) > 0.03:
                    f_md_path = f_md_path3 
            if do_md == False:
                return None, None
            else:
                return top_dir, f_md_path
        else:
            print(f'Not exist: {top_dir}')
            return None, None
                
    
    # 考虑到存储问题，我们只保存md模拟最后的gro文件，此外，我们还需采用gmx energy 分析数据
    # 用于判断最终的gro文件是平衡结构
    with Pool(20) as pool:
        results = pool.map(process_mol_dir, [i for i in range(mol_num)])
    mol_dirs = []; md_commands = []
    for ii in results:
        if ii[0] is not None:
            mol_dirs.append(ii[0])
            md_commands.append(ii[1])

    task_set = []; md_command_sets = []
    for ii in range(2000):
        task_set.append([])
        md_command_sets.append([])
    for ii, f_dir in enumerate(mol_dirs):
        task_set[ii%2000].append(f_dir)
        md_command_sets[ii%2000].append(md_commands[ii])
    
    cwd = os.path.abspath(os.getcwd())
    for ii in range(2000):
        commands = []
        commands.append('module purge')
        commands.append('source /public/home/chengz/apprepo/gromacs_threadMPI/2023.2-dtk23.10_hpcx241/scripts/env.sh')
        for idx, jj in enumerate(task_set[ii]):
            mol_name = os.path.basename(os.path.dirname(jj))
            local_path = os.path.abspath(jj)
            tmp_path = f'/tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}'
            f_md_path_final = md_command_sets[ii][idx]
            commands.append(f'mkdir -p /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {local_path}/min.gro /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {f_min_path} /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {f_md_path_final} /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}/md.mdp')
            commands.append(f'cp {local_path}/mol.top /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {local_path}/mol.itp /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {py_path} /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {local_path}/mol_checked.sdf /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cd /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'gmx grompp -f md.mdp -p mol.top -c min.gro -o md.tpr > 0.out 2>&1')            
            # 可以将输出保存>1.out然后check 
            commands.append('sleep 1')
            commands.append(f'gmx mdrun -gpu_id 0 -ntmpi 1 -ntomp 7 -nb gpu -pme gpu -deffnm md > tmp.out 2>&1')
            commands.append('sleep 1')
            commands.append(f'echo -e "Density\\n" | gmx energy -f md.edr -o energy.xvg > 1.out 2>&1')
            commands.append('sleep 1')
            commands.append(f'echo -e "0\\n0\\n" | gmx trjconv -f md.xtc -s md.tpr -dump 999999 -o lst.gro -pbc mol -center > 2.out 2>&1')
            #commands.append(f'gmx editconf -f lst.gro -o lst.xyz')
            commands.append('sleep 1')
            commands.append(f'python check_min.py')
            # 可能需要检查energy_1.xvg, 如果密度变化过于剧烈。。。第三轮md模拟平衡，还是很小直接放弃
            commands.append(f'cp {tmp_path}/energy.xvg {local_path}/energy_2.xvg')
            commands.append(f'cp {tmp_path}/lst.gro {local_path}')
            commands.append(f'cp {tmp_path}/res.xvg {local_path}')
            commands.append(f'cp {tmp_path}/md.gro {local_path}/min.gro')
            #Gcommands.append(f'rm {local_path}/mol.gro')
            commands.append(f'cd {cwd}')
            commands.append(f'rm -rf /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append('sleep 1')
        generate_slurm_gpu(commands, ii, 8, 'md')



elif opera == 'runmin_inbatch':
    # we have two npz files, mol_for_md.npz & mrtadf_osc.npz 
    npz_dir = './forcefield'
    data = np.load(f'{npz_dir}/mol_for_md.npz',allow_pickle=True)
    # 控制文件位置
    f_min_path = '/public/home/chengz/photomat/min.mdp'
    f_md_path = '/public/home/chengz/photomat/md.mdp'
    py_path = '/public/home/chengz/photomat/check_min.py'
    
    #npz_dir = './forcefield_mrtadf'
    #data = np.load(f'{npz_dir}/mrtadf_osc.npz',allow_pickle=True)
    
    mol_num = len(data['smi'])
    mol_dirs = []
    for ii in range(mol_num):
        top_dir = os.path.join(f'{npz_dir}/mol_{ii}','final')
        if os.path.exists(os.path.join(top_dir,'mol.gro')):
            mol_dirs.append(top_dir)

    # 考虑到存储问题，我们只保存md模拟最后的gro文件，此外，我们还需采用gmx energy 分析数据
    # 用于判断最终的gro文件是平衡结构
    task_set = []
    for ii in range(500):
        task_set.append([])
    for ii, f_dir in enumerate(mol_dirs):
        task_set[ii%500].append(f_dir)
    
    cwd = os.path.abspath(os.getcwd())
    for ii in range(500):
        commands = []
        commands.append('module purge')
        commands.append('source /public/home/chengz/apprepo/gromacs/2023-gcc930_intelmpi17/scripts/env.sh')
        for jj in task_set[ii]:
            if os.path.exists(os.path.join(jj,'energy.xvg')):
                with open(os.path.join(jj,'energy.xvg'),'r') as f:
                    lines = f.readlines()
                if 'succ' in lines[0]:
                    continue
                else:
                    print(f'Error: {jj}')
            else:
                print(f'Not exist: {jj}')
            mol_name = os.path.basename(os.path.dirname(jj))
            local_path = os.path.abspath(jj)
            tmp_path = f'/tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}'
            commands.append(f'mkdir -p /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {local_path}/mol.gro /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {f_min_path} /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {f_md_path} /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {local_path}/mol.top /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {local_path}/mol.itp /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {py_path} /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cp {local_path}/mol_checked.sdf /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'cd /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
            commands.append(f'gmx_mpi grompp -f min.mdp -p mol.top -c mol.gro -o min.tpr')
            commands.append(f'mpirun -bootstrap ssh -np 8 gmx_mpi mdrun -deffnm min')
            #commands.append(f'gmx mdrun -gpu_id 0 -ntmpi 1 -ntomp 7 -nb gpu -deffnm min')
            # 将min结构导出
            # md模拟过程，我们发现min还是在cpu跑快，因为gpu用不上
            # md可能分两步，压缩盒子&平衡结构, 若密度稳定，压缩系数变小，仍很小就维持压缩系数压
            commands.append(f'echo -e "0\\n0\\n" | gmx_mpi trjconv -s min.tpr -f min.trr -o check.gro -pbc mol -center')
            commands.append(f'python check_min.py')
            commands.append(f'cp {tmp_path}/min.gro {local_path}')
            commands.append(f'cp {tmp_path}/energy.xvg {local_path}')
            commands.append(f'cd {cwd}')
            commands.append(f'rm -rf /tmp/scrtch/chengz/gro_$SLURM_JOB_ID/{mol_name}')
        generate_slurm(commands, ii, 8, 'md')

