gmx grompp -f md_0.mdp -p mol_oscff.top -c mol.gro -o md.tpr > 0.out 2>&1
gmx mdrun -gpu_id 0 -ntmpi 1 -ntomp 7 -nb gpu -pme gpu -deffnm md > tmp.out 2>&1
