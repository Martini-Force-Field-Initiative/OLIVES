#!/bin/env bash
#Run this script with command "nohup bash build.sh </dev/null >&out&"
source /home/au540368/software/python3/environments/defenv/bin/activate

martinize2 -f 2OOB.pdb -x 2OOB_CG.pdb -o 2OOB_martinize2.top -cys auto -scfix
rm 2OOB_martinize2.top

gmx editconf -f 2OOB_CG.pdb -o 2OOB_CG.gro -bt dodecahedron -d 2.5 -c

gmx make_ndx -f 2OOB_CG.pdb -o sys.ndx <<< 'del 0-19
chain A
chain B
q
'
echo chA | gmx trjconv -f 2OOB_CG.pdb -s 2OOB_CG.pdb -o 2OOB_CG_0.pdb -n sys.ndx
echo chB | gmx trjconv -f 2OOB_CG.pdb -s 2OOB_CG.pdb -o 2OOB_CG_1.pdb -n sys.ndx

#Apply the OLIVES model	
python3 OLIVES_v1.0_M3.0.0.py -c 2OOB_CG_0.pdb -i molecule_0.itp 
python3 OLIVES_v1.0_M3.0.0.py -c 2OOB_CG_1.pdb -i molecule_1.itp

gmx make_ndx -f 2OOB_CG_0.pdb -o BB_0.ndx <<< 'del 0-19
a BB
q
'
echo 0 | gmx genrestr -f 2OOB_CG_0.pdb -n BB_0.ndx -o posre_BB_0.itp

gmx make_ndx -f 2OOB_CG_1.pdb -o BB_1.ndx <<< 'del 0-19
a BB
q
'
echo 0 | gmx genrestr -f 2OOB_CG_1.pdb -n BB_1.ndx -o posre_BB_1.itp

cp 2OOB.top 2OOB_solv.top
gmx solvate -cp 2OOB_CG.gro -cs W.gro -p 2OOB_solv.top -o 2OOB_CG.gro

gmx grompp -f ions.mdp -c 2OOB_CG.gro -p 2OOB_solv.top -o ions.tpr
echo 13 | gmx genion -s ions.tpr -o 2OOB_CG.gro -p 2OOB_solv.top -pname NA -nname CL -neutral -conc 0.15

#Minimization
gmx grompp -f min.mdp -c 2OOB_CG.gro -r 2OOB_CG.gro -p 2OOB_solv.top -o min.tpr
gmx mdrun -deffnm min
wait

rm \#*

