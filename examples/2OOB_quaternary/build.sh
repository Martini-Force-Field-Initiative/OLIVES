#!/bin/env bash

martinize2 -f 2OOB.pdb -x 2OOB_CG.pdb -o 2OOB_martinize2.top -cys auto -scfix -merge A,B
rm 2OOB_martinize2.top
#See 2OOB.top

#Apply the OLIVES model	
python3 ../../OLIVES_v1.3_M3.0.0.py -c 2OOB_CG.pdb -i molecule_0.itp 

gmx editconf -f 2OOB_CG.pdb -o 2OOB_CG.gro -bt dodecahedron -d 2.5 -c

gmx make_ndx -f 2OOB_CG.gro -o BB.ndx <<< 'del 0-19
a BB
q
'

gmx genrestr -f 2OOB_CG.gro -n BB.ndx -o posre_BB.itp

cp 2OOB.top 2OOB_solv.top
gmx solvate -cp 2OOB_CG.gro -cs W.gro -p 2OOB_solv.top -o 2OOB_CG.gro

gmx grompp -f ions.mdp -c 2OOB_CG.gro -p 2OOB_solv.top -o ions.tpr
echo W | gmx genion -s ions.tpr -o 2OOB_CG.gro -p 2OOB_solv.top -pname NA -nname CL -neutral -conc 0.15

#Minimization
gmx grompp -f min.mdp -c 2OOB_CG.gro -r 2OOB_CG.gro -p 2OOB_solv.top -o min.tpr
gmx mdrun -deffnm min
wait

rm \#*

