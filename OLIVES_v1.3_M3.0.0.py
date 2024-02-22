#!/usr/bin/env python
print(" o o o o o o o o o ")
print("OLIVES_v1.3_M3.0.0.py is written by Kasper Busk Pedersen, February 22th 2024.")
print("If you use this script in your work, please cite Pedersen et al. DOI:10.26434/chemrxiv-2023-6d61w")

import sys
import math
import argparse
import numpy as np
import mdtraj as md
import networkx as nx

def user_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', help='File containing the coarse-grained structure(s) of the protein in pdb format. Multiple conformations can be supplied separated using comma: conf1.pdb,conf2.pdb,conf3.pdb. The conformations must have the exact same topology.')
    parser.add_argument('-i', help='File containing the protein topology e.g. molecule_0.itp from martinize2.')
    parser.add_argument('--ss_cutoff', type=float, default=0.55, help='Cutoff distance for generation of the secondary network (and quaternary between BB beads) (default=0.55).')
    parser.add_argument('--ts_cutoff', type=float, default=0.55, help='Cutoff distance for generation of the tertiary network (and quaternary between non BB-BB pairs) (default=0.55).')
    parser.add_argument('--ss_scaling', type=float, default=1.0, help='Scaling of the secondary structure OLIVES bonds (default=1.0).')
    parser.add_argument('--ts_scaling', type=float, default=1.0, help='Scaling of the tertiary structure OLIVES bonds (default=1.0).')
    parser.add_argument('--qs_scaling', type=str, default='{}', help='Scaling of the hydrogen bonds between chain indices (chains must have different names in the pdb file format). Format is a dictionary of chain index tuples with associated scaling value. E.g. in a protein with chain A,B,C the input {(0,1):0.7,(0,2):0.5,(1,2):0.2} will scale quaternary bonds between chain A and B by 0.7, A and C by 0.5, and B and C by 0.2. The tuple must be sorted.')
    parser.add_argument('--unique_pair_scaling', type=str, default="1.0", help='Multistate mode: Scaling of the unique contacts in each conformation, provided as a comma separated sting e.g. "0.5,0.75". Shared contacts between conformations have their distances averaged.')
    parser.add_argument('--extend_itp', type=int, default=True, help='Extend the protein topology with the Go-like model (default=True).')
    parser.add_argument('--write_separate_itp', type=bool, default=False, help='Writes the Go-like model as separate itp files to be included in the .top (default=False).')
    parser.add_argument('--write_vmd_itp', type=bool, default=False, help='Writes an itp with the Go-like bonds as harmonic bonds for visualization with VMD or use OLIVES as an elastic network (default=False).')
    parser.add_argument('--write_bond_information_file', type=bool, default=False, help='Writes a datafile with the bead indices for each contact pair. Does set analysis in the case of multiple conformations (default=False).')
    parser.add_argument('--silent', type=bool, default=False, help='sshhh (default=False)')
    args = parser.parse_args()
    return args

##### CHECK INPUT #####

args = user_input()
input_conformations = list(args.c.split(","))
itp_CG = args.i
secondary_cutoff = args.ss_cutoff  #[nm] - Distance cutoff for defining a hbond 
tertiary_cutoff = args.ts_cutoff  #[nm] - Distance cutoff for defining a hbond  
secondary_energy_scaling = args.ss_scaling  #Is multiplied with the relative hbond energies 
tertiary_energy_scaling = args.ts_scaling  #Is multiplied with the relative hbond energies
quaternary_energy_scaling = eval(args.qs_scaling)  #Is multiplied with the relative hbond energies of bonds between chains - will overwrite any scaling of secodnary and tertiary structure 
unique_pair_scaling = [float(i) for i in list(args.unique_pair_scaling.split(","))]  #[kJ/mol] - Is multiplied with the relative hbond energies of conformational unique HBs when providing multiply conformations
harm_k = 500 #Used for visualization, but could be set if you want OLIVES as an elastic network
silent = args.silent

#Load the conformations
pdbs_CG = [md.load(pdb_CG) for pdb_CG in input_conformations]

#Check if the topologies match
natoms = [pdb.n_atoms for pdb in pdbs_CG]
nresidues = [pdb.n_residues for pdb in pdbs_CG]
nchains = [pdb.n_chains for pdb in pdbs_CG]
if not np.sum(natoms) == len(natoms)*np.min(natoms):
    raise ValueError('Your input conformations do not have the same number of beads. Please check your topology.')
if not np.sum(nresidues) == len(nresidues)*np.min(nresidues):
    raise ValueError('Your input conformations do not have the same number of residues. Please check your topology.')
if not np.sum(nchains) == len(nchains)*np.min(nchains):
    raise ValueError('Your input conformations do not have the same number of chains. Please check your topology.')

#If quaternary structure scaling is provided check that the dictionary is the correct size (does not correct user errors in tuples)
if bool(quaternary_energy_scaling):
    if not int(math.factorial(nchains[0])/2) == len(quaternary_energy_scaling):
        raise ValueError('You have provided a dictionary of tuples of with quaternary structure scaling values but the lenth of the dictionary is not factorial(nchains)/2. Please provide a dictionary of the correct size. Note that we do not check if you are creating the correct tuples. See format explaination in help with "-h".')

#Check if the number of conformations match in the number of scaling values
if not len(input_conformations) == len(unique_pair_scaling):
    raise ValueError('The number of input conformations do not match the number of scaling values provided by the --unique_pair_scaling flag.')

##### DEFINITION OF THE MODEL #####

# Format:
# [[HBA,HBD], [HBA_type,HBD_type]], 1=yes 0=no
# IMA: imidazole HBA, AMA: Amide HBA, KEA: kentone HBA, PIA: pi HBA, IMD: imidazole HBD, AMD: amide HBD, IND: indole HBD, HYD: hydroxyl HBD
#There are some missing values in the energy matrix:
#hydroxyl acceptors: approximated by ketone acceptor energy
#lysine and arginine donors: approximated by the imidazole donor energy
#cysteine (donor,acceptor) and methionine (acceptor) are not represented: approximated by the hydroxyl donor and ketone acceptor
hbond_dict = {"GLY":{"BB":[[1,1],["AMA","AMD"]]},
              "ALA":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[0,0],["None","None"]]},
              "CYS":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[1,1],["KEA","HYD"]]},
              "CYP":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[1,0],["KEA","None"]],"C1":[[0,0],["None","None"]],"C2":[[0,0],["None","None"]],"C3":[[0,0],["None","None"]],"C4":[[0,0],["None","None"]]},  #palmitoylation
              "VAL":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[0,0],["None","None"]]},
              "LEU":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[0,0],["None","None"]]},
              "ILE":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[0,0],["None","None"]]},
              "MET":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[1,0],["KEA","None"]]},
              "PRO":{"BB":[[1,0],["AMA","None"]],"SC1":[[0,0],["None","None"]]},
              "ASN":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[1,1],["AMA","AMD"]]}, 
              "GLN":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[1,1],["AMA","AMD"]]}, 
              "ASP":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[1,0],["KEA","None"]]},
              "ASPP":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[1,1],["KEA","HYD"]]},
              "ASH":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[1,1],["KEA","HYD"]]},
              "GLU":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[1,0],["KEA","None"]]},
              "GLUP":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[1,1],["KEA","HYD"]]},
              "GLH":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[1,1],["KEA","HYD"]]},
              "THR":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[1,1],["KEA","HYD"]]},
              "SER":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[1,1],["KEA","HYD"]]},
              "LYS":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[0,0],["None","None"]],"SC2":[[0,1],["None","IMD"]]},
              "LSN":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[0,0],["None","None"]],"SC2":[[1,1],["IMA","IMD"]]},
              "LYN":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[0,0],["None","None"]],"SC2":[[1,1],["IMA","IMD"]]},
              "ARG":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[0,0],["None","None"]],"SC2":[[0,1],["None","IMD"]]}, 
              "HIS":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[0,0],["None","None"]],"SC2":[[0,1],["None","IMD"]],"SC3":[[1,0],["IMA","None"]]}, #assumed epsilon tautomer, but you should correct your pdb naming to HSE/HSD/HSP or HIE/HID/HIP naming (you should care about histidine tautomers)
              "HIE":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[0,0],["None","None"]],"SC2":[[0,1],["None","IMD"]],"SC3":[[1,0],["IMA","None"]]}, 
              "HSE":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[0,0],["None","None"]],"SC2":[[0,1],["None","IMD"]],"SC3":[[1,0],["IMA","None"]]},
              "HID":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[0,0],["None","None"]],"SC2":[[1,0],["IMA","None"]],"SC3":[[0,1],["None","IMD"]]},
              "HSD":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[0,0],["None","None"]],"SC2":[[1,0],["IMA","None"]],"SC3":[[0,1],["None","IMD"]]},
              "HSP":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[0,0],["None","None"]],"SC2":[[0,1],["None","IMD"]],"SC3":[[0,1],["None","IMD"]]},
              "HIP":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[0,0],["None","None"]],"SC2":[[0,1],["None","IMD"]],"SC3":[[0,1],["None","IMD"]]},
              "PHE":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[0,0],["None","None"]],"SC2":[[1,0],["PIA","None"]],"SC3":[[1,0],["PIA","None"]]},
              "TYR":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[0,0],["None","None"]],"SC2":[[1,0],["PIA","None"]],"SC3":[[1,0],["PIA","None"]],"SC4":[[1,1],["KEA","HYD"]]},
              "TRP":{"BB":[[1,1],["AMA","AMD"]],"SC1":[[0,0],["None","None"]],"SC2":[[0,1],["None","IND"]],"SC3":[[0,0],["None","None"]],"SC4":[[1,0],["PIA","None"]],"SC5":[[1,0],["PIA","None"]]}}


##We wont fit each HB energy individually, instead we approximate the relation between them using ab initio results and give an option to scale the ladder
#Sparse energy matrix:
energy_dict = {"IMA-AMD":6.55*4.184,  #4.184 converts from kcal/mol -> kJ/mol
               "IMA-HYD":6.24*4.184,
               "IMA-IND":7.11*4.184,
               "IMA-IMD":7.96*4.184,
               "AMA-AMD":5.44*4.184,           #<-In the secondary structure model we exclude BB-BB (P2-P2) interactions, because the sigma for this interaction is larger than the structural distance between 
               "AMA-HYD":4.51*4.184,           #most backbone beads in a proteins (one of the main reasons why the v3.0.0 mapping doesnt work for ss). To be consistent with the tertiary model that doesnt use exclusiosn
               "AMA-IND":6.27*4.184,           #we add back in the interaction energy of the P2-P2 LJ later in the function "combine and format potentials. The reason we dont use exclusion in the tertiary model
               "AMA-IMD":6.92*4.184,           #is that GROMACS doesnt support intermolecular exclusions and we want the tertiary model to work also for quaternary structure.
               "KEA-AMD":4.06*4.184,          
               "KEA-HYD":3.54*4.184,           
               "KEA-IND":5.12*4.184,
               "KEA-IMD":5.78*4.184,
               "PIA-AMD":22.2329/2,   #Divide by 2 because we split the pi interaction over the 2 outmost beads - important for the ring conformation
               "PIA-HYD":np.mean([19.7648,21.7831])/2,  #here we mean the two results from the paper
               "PIA-IND":8.776/2,
               "PIA-IMD":25.771/2}

#Relative energies from: 
#Ming-Hong Hao, Theoretical Calculation of Hydrogen-Bonding Strength
#for Drug Molecules (Table 2)
#https://doi.org/10.1021/ct0600262

#HBD-pi are from:
#Du et al., Theoretical study on the polar hydrogen-π (Hp-π)
#interactions between protein side chains (Table 2+3)
#doi: 10.1186/1752-153X-7-92

#Hydrogen bonds with Sulfur are approximately the same strength as their oxygen counterparts:
#Mundlapati, V. Rao, et al. "Critical assessment of the strength of hydrogen bonds between the sulfur atom of methionine/cysteine and backbone amides in proteins." The journal of physical chemistry letters 6.8 (2015): 1385-1389.
#https://doi.org/10.1021/acs.jpclett.5b00491
#"The E_DA were found to be 53.9
#and 37.5 kJ/mol for amide N-H···S-methionine and amide
#N-H···S-cysteine H-bonds, respectively; very similar to those
#observed for simple model peptides. This justifies our claim
#that “amide N-H···S H-bonds in proteins and bio-molecules
#are equally strong H-bonds as their oxygen counterpart”.

##### FUNCTIONS #####
def knowledge_based_checks(cut_pairs,cut_dists,filters,silent):
    #Unpack filters
    a_ndx_to_bead_name = filters[0]
    a_ndx_to_res_name = filters[1]
    a_ndx_to_res_ndx = filters[2]

    #Detect cys bridges based on a distance criterion of 0.3 nm 
    cys_bridges = []
    for i,pair in enumerate(cut_pairs):
        if a_ndx_to_res_name[pair[0]] == 'CYS' and a_ndx_to_res_name[pair[1]] == 'CYS':
            if not a_ndx_to_bead_name[pair[0]] == 'BB' or a_ndx_to_bead_name[pair[1]] == 'BB':
                if cut_dists[i] < 0.3:
                    cys_bridges.append(pair)
    print("Detected {} disulfide bridges".format(len(cys_bridges)))
    
    pairs_secondary = []
    dists_secondary = []
    pairs_tertiary = []
    dists_tertiary = []
    
    for i,pair in enumerate(cut_pairs):
        if not silent:
            #Progress bar
            total = len(cut_pairs)
            bar_length = 30
            percent = 100.0*i/total
            sys.stdout.write('\r')
            sys.stdout.write("Knowledge based checks: [{:{}}] {:>3}%".format('='*int(percent/(100.0/bar_length)),bar_length, int(percent)))
            sys.stdout.flush()

        #Sanity checks
        if a_ndx_to_bead_name[pair[0]] == 'CA' or a_ndx_to_bead_name[pair[1]] == 'CA':
            raise ValueError('It looks like you have an atomistic structure or used -govs-includes in martinize2 which was not expected.')
        if a_ndx_to_res_ndx[pair[0]] == a_ndx_to_res_ndx[pair[1]]:
            continue  #exclude intra-residue hbonds
        
        #Exclusion rules for the secondary network
        if a_ndx_to_bead_name[pair[0]] == 'BB' and a_ndx_to_bead_name[pair[1]] == 'BB':
            #Rules for the secondary network
            if a_ndx_to_res_ndx[pair[1]] == (a_ndx_to_res_ndx[pair[0]]+1):
                continue # exclude BB <-> BB+1 hbonds
            elif a_ndx_to_res_ndx[pair[1]] == (a_ndx_to_res_ndx[pair[0]]+2):
                continue # exclude BB <-> BB+2 hbonds 
            else:
                pairs_secondary.append(pair)
                dists_secondary.append(cut_dists[i])
                continue

        #Exclusion rules for tertiary network
        if a_ndx_to_bead_name[pair[0]] == 'BB' or a_ndx_to_bead_name[pair[1]] == 'BB':
            if a_ndx_to_res_ndx[pair[0]]+1 == (a_ndx_to_res_ndx[pair[1]]):
                continue # exclude BB <-> sidechain+1 hbonds
       
        #Deal with cys bridges
        cys_ignore = False
        for b in cys_bridges:
            #Exclude the sulfur atoms 
            if pair[0] == b[0] and pair[1] == b[1]:
                cys_ignore = True
            
            #Exclude backbone to sulfur bonds within the bridge
            if a_ndx_to_res_ndx[pair[0]] == a_ndx_to_res_ndx[b[0]]:
                if a_ndx_to_res_ndx[pair[1]] == a_ndx_to_res_ndx[b[1]]:
                    cys_ignore = True
        
            if a_ndx_to_res_ndx[pair[1]] == a_ndx_to_res_ndx[b[1]]:
                if a_ndx_to_res_ndx[pair[0]] == a_ndx_to_res_ndx[b[0]]:
                    cys_ignore = True
            
            if a_ndx_to_res_ndx[pair[0]] == a_ndx_to_res_ndx[b[1]]:
                if a_ndx_to_res_ndx[pair[1]] == a_ndx_to_res_ndx[b[0]]:
                    cys_ignore = True
            
            if a_ndx_to_res_ndx[pair[1]] == a_ndx_to_res_ndx[b[0]]:
                if a_ndx_to_res_ndx[pair[0]] == a_ndx_to_res_ndx[b[1]]:
                    cys_ignore = True

        if cys_ignore:
            continue
        
        pairs_tertiary.append(pair)
        dists_tertiary.append(cut_dists[i])
        
    if not silent:
        #Save progress bar 
        percent = 100
        sys.stdout.write('\r')
        sys.stdout.write("Knowledge based checks: [{:{}}] {:>3}%".format('='*int(percent/(100.0/bar_length)),bar_length, int(percent)))
        sys.stdout.flush()
        sys.stdout.write('\n')
    return np.array(pairs_secondary),np.array(dists_secondary),np.array(pairs_tertiary),np.array(dists_tertiary)

def check_HB_pairs(pairs,dists,hbond_dict,energy_dict,filters,silent):
    #Unpack filters
    a_ndx_to_bead_name = filters[0]
    a_ndx_to_res_name = filters[1]
    a_ndx_to_res_ndx = filters[2]
    
    #Now we do HBD<->HBA based checks
    checked_pairs = []
    checked_pairs_dists_energies = []
    for i,pair in enumerate(pairs):
        if not silent:
            #Progress bar
            total = len(pairs)
            bar_length = 30
            percent = 100.0*i/total
            sys.stdout.write('\r')
            sys.stdout.write("Check HB pairs: [{:{}}] {:>3}%".format('='*int(percent/(100.0/bar_length)),bar_length, int(percent)))
            sys.stdout.flush()
        
        #Get information about the pair
        p0_beadname = a_ndx_to_bead_name[pair[0]]
        p0_resname = a_ndx_to_res_name[pair[0]]
        p0_bead_dict = hbond_dict[p0_resname]
        p0_hbond_list = p0_bead_dict[p0_beadname]
        
        p1_beadname = a_ndx_to_bead_name[pair[1]]
        p1_resname = a_ndx_to_res_name[pair[1]]
        p1_bead_dict = hbond_dict[p1_resname]
        p1_hbond_list = p1_bead_dict[p1_beadname]
        
        #Now figure out if the pair is HBA<->HBD
        
        if p0_hbond_list[0][0] and p1_hbond_list[0][1]: #First bead is HBA and second is HBD
            if p0_hbond_list[0][1] and p1_hbond_list[0][0]:  #First bead is also HBD and second is also HBA
                HBA_HBD_partners = p0_hbond_list[1][0]+"-"+p1_hbond_list[1][1]
                HBD_HBA_partners = p1_hbond_list[1][0]+"-"+p0_hbond_list[1][1]
                HBA_HBD_energy = energy_dict[HBA_HBD_partners]
                HBD_HBA_energy = energy_dict[HBD_HBA_partners]
                HB_energy = np.max([HBA_HBD_energy,HBD_HBA_energy]) #Take the strongest possible hbond which could be formed by the pair
                checked_pairs.append((pair[0],pair[1]))
                checked_pairs_dists_energies.append([(pair[0],pair[1]),dists[i],HB_energy])  
                continue
    
            HB_partners = p0_hbond_list[1][0]+"-"+p1_hbond_list[1][1]
            HB_energy = energy_dict[HB_partners]
            checked_pairs.append((pair[0],pair[1]))
            checked_pairs_dists_energies.append([(pair[0],pair[1]),dists[i],HB_energy])  
            continue
        
        if p0_hbond_list[0][1] and p1_hbond_list[0][0]: #First bead is HBD and second is HBA
            HB_partners = p1_hbond_list[1][0]+"-"+p0_hbond_list[1][1]
            HB_energy = energy_dict[HB_partners]
            checked_pairs.append((pair[0],pair[1]))
            checked_pairs_dists_energies.append([(pair[0],pair[1]),dists[i],HB_energy])  
            continue
    
    if not silent:
        #Save progress bar 
        percent = 100
        sys.stdout.write('\r')
        sys.stdout.write("Check HB pairs: [{:{}}] {:>3}%".format('='*int(percent/(100.0/bar_length)),bar_length, int(percent)))
        sys.stdout.flush()
        sys.stdout.write('\n')
    return checked_pairs,checked_pairs_dists_energies

def get_pair_set_information(input_conformations,all_secondary_pairs,all_tertiary_pairs):
    #Output information on OLIVES pair sets
    info_secondary_pairs = []
    info_tertiary_pairs = []
    unique_secondary_pairs_for_scaling = []
    unique_tertiary_pairs_for_scaling = []
    if len(input_conformations) == 1:
        #all OLIVES pairs
        info_secondary_pairs.append('; pairs in {}'.format(input_conformations[0]).split())
        info_secondary_pairs.append(str(sorted([(p[0]+1,p[1]+1) for p in all_secondary_pairs[0]])).split())
        
        info_tertiary_pairs.append('; pairs in {}'.format(input_conformations[0]).split())
        info_tertiary_pairs.append(str(sorted([(p[0]+1,p[1]+1) for p in all_tertiary_pairs[0]])).split())
    
    elif len(input_conformations) == 2:
        #all OLIVES pairs
        info_secondary_pairs.append('; pairs in {}'.format(input_conformations[0]).split())
        info_secondary_pairs.append(str(sorted([(p[0]+1,p[1]+1) for p in all_secondary_pairs[0]])).split())
        info_secondary_pairs.append('; pairs in {}'.format(input_conformations[1]).split())
        info_secondary_pairs.append(str(sorted([(p[0]+1,p[1]+1) for p in all_secondary_pairs[1]])).split())
        
        info_tertiary_pairs.append('; pairs in {}'.format(input_conformations[0]).split())
        info_tertiary_pairs.append(str(sorted([(p[0]+1,p[1]+1) for p in all_tertiary_pairs[0]])).split())
        info_tertiary_pairs.append('; pairs in {}'.format(input_conformations[1]).split())
        info_tertiary_pairs.append(str(sorted([(p[0]+1,p[1]+1) for p in all_tertiary_pairs[1]])).split())
        
        #pairs unique to conformations
        unique_pairs_conf1 = set(all_secondary_pairs[0])-set(all_secondary_pairs[1])
        info_secondary_pairs.append('; pairs unique to {}'.format(input_conformations[0]).split())
        info_secondary_pairs.append(str(sorted([(p[0]+1,p[1]+1) for p in unique_pairs_conf1])).split())
        unique_secondary_pairs_for_scaling.append(sorted([(p[0],p[1]) for p in unique_pairs_conf1]))
        
        unique_pairs_conf2 = set(all_secondary_pairs[1])-set(all_secondary_pairs[0])
        info_secondary_pairs.append('; pairs unique to {}'.format(input_conformations[1]).split())
        info_secondary_pairs.append(str(sorted([(p[0]+1,p[1]+1) for p in unique_pairs_conf2])).split())
        unique_secondary_pairs_for_scaling.append(sorted([(p[0],p[1]) for p in unique_pairs_conf2]))
        
        unique_pairs_conf1 = set(all_tertiary_pairs[0])-set(all_tertiary_pairs[1])
        info_tertiary_pairs.append('; pairs unique to {}'.format(input_conformations[0]).split())
        info_tertiary_pairs.append(str(sorted([(p[0]+1,p[1]+1) for p in unique_pairs_conf1])).split())
        unique_tertiary_pairs_for_scaling.append(sorted([(p[0],p[1]) for p in unique_pairs_conf1]))
        
        unique_pairs_conf2 = set(all_tertiary_pairs[1])-set(all_tertiary_pairs[0])
        info_tertiary_pairs.append('; pairs unique to {}'.format(input_conformations[1]).split())
        info_tertiary_pairs.append(str(sorted([(p[0]+1,p[1]+1) for p in unique_pairs_conf2])).split())
        unique_tertiary_pairs_for_scaling.append(sorted([(p[0],p[1]) for p in unique_pairs_conf2]))
        
        #intersection 
        intersection = set(all_secondary_pairs[0]) & set(all_secondary_pairs[1])
        info_secondary_pairs.append('; intersection of {}'.format(str(input_conformations)).split())
        info_secondary_pairs.append(str(sorted([(p[0]+1,p[1]+1) for p in intersection])).split())
        
        intersection = set(all_tertiary_pairs[0]) & set(all_tertiary_pairs[1])
        info_tertiary_pairs.append('; intersection of {}'.format(str(input_conformations)).split())
        info_tertiary_pairs.append(str(sorted([(p[0]+1,p[1]+1) for p in intersection])).split())

    elif len(input_conformations) > 2:
        conf_ndx = range(len(input_conformations))
        for c in conf_ndx:
            #all OLIVES pairs
            info_secondary_pairs.append('; pairs in {}'.format(input_conformations[c]).split())
            info_secondary_pairs.append(str(sorted([(p[0]+1,p[1]+1) for p in all_secondary_pairs[c]])).split())
    
            info_tertiary_pairs.append('; pairs in {}'.format(input_conformations[c]).split())
            info_tertiary_pairs.append(str(sorted([(p[0]+1,p[1]+1) for p in all_tertiary_pairs[c]])).split())
    
        for c in conf_ndx:
            #pairs unique to conformations
            conf_set = set(all_secondary_pairs[c])
            other_confs = [all_secondary_pairs[i] for i in conf_ndx if not i==c]
            other_confs_set = set([item for sublist in other_confs for item in sublist])
            info_secondary_pairs.append('; pairs unique to {}'.format(input_conformations[c]).split())
            unique_pairs_conf = conf_set - other_confs_set
            info_secondary_pairs.append(str(sorted([(p[0]+1,p[1]+1) for p in unique_pairs_conf])).split())
            unique_secondary_pairs_for_scaling.append(sorted([(p[0],p[1]) for p in unique_pairs_conf]))
            
            conf_set = set(all_tertiary_pairs[c])
            other_confs = [all_tertiary_pairs[i] for i in conf_ndx if not i==c]
            other_confs_set = set([item for sublist in other_confs for item in sublist])
            info_tertiary_pairs.append('; pairs unique to {}'.format(input_conformations[c]).split())
            unique_pairs_conf = conf_set - other_confs_set
            info_tertiary_pairs.append(str(sorted([(p[0]+1,p[1]+1) for p in unique_pairs_conf])).split())
            unique_tertiary_pairs_for_scaling.append(sorted([(p[0],p[1]) for p in unique_pairs_conf]))
        
        #intersection 
        set_list = [set(l) for l in all_secondary_pairs]
        intersection = set.intersection(*set_list)
        info_secondary_pairs.append('; intersection of {}'.format(str(input_conformations)).split())
        info_secondary_pairs.append(str(sorted([(p[0]+1,p[1]+1) for p in intersection])).split())
        
        set_list = [set(l) for l in all_tertiary_pairs]
        intersection = set.intersection(*set_list)
        info_tertiary_pairs.append('; intersection of {}'.format(str(input_conformations)).split())
        info_tertiary_pairs.append(str(sorted([(p[0]+1,p[1]+1) for p in intersection])).split())
    else:
        pass
    return info_secondary_pairs, info_tertiary_pairs, unique_secondary_pairs_for_scaling, unique_tertiary_pairs_for_scaling

def construct_pair_multiples_dict(all_pairs,all_pairs_dists_energies):
    #Helper function that puts intersection pairs into a dict. Distances of interaction pairs are later averaged
    pairs_dict = {}
    for s,structure_pairs in enumerate(all_pairs):
        #Construct dict
        for p,pair in enumerate(structure_pairs):
            current_entries = pairs_dict.keys()
            if pair in current_entries:
                currect_entry = pairs_dict[pair]
                pairs_dict.update({pair: currect_entry+[[all_pairs_dists_energies[s][p][1],all_pairs_dists_energies[s][p][2]]]})
            else:
                pairs_dict.update({pair: [[all_pairs_dists_energies[s][p][1],all_pairs_dists_energies[s][p][2]]]})
    return pairs_dict

def combine_and_format_potentials(pairs_dict,networktype,unique_pairs,unique_pair_scaling,energy_scaling,quaternary_energy_scaling,filters):
    #Formats the potentials before inserting into .itp
    #GROMACS function types
    a_ndx_to_chain_ndx = filters[3]
    harm_function_type = 1
    LJ_function_type = 1
    
    harm_pairs = []
    LJ_pairs = []
    excluded_pairs = []
    quaternary_pairs_to_chain = {}
    quaternary_pair_info = []

    for pair, dist_energy in pairs_dict.items():
        #Check if the pair is a quaternary bond
        if not a_ndx_to_chain_ndx[pair[0]] == a_ndx_to_chain_ndx[pair[1]]:
            chain_one = a_ndx_to_chain_ndx[pair[0]]
            chain_two = a_ndx_to_chain_ndx[pair[1]]
            quaternary_pairs_to_chain.update({pair:(chain_one,chain_two)})
            quaternary_pair_info.append([str((pair[0]+1,pair[1]+1)),str((chain_one,chain_two))])


        dists = np.array(dist_energy)[:,0]
        if len(unique_pair_scaling) > 1:
            for i,conf in enumerate(unique_pairs):
                if pair in conf:
                    HB_energy = np.array(dist_energy)[0,1]*energy_scaling*unique_pair_scaling[i]

                    #Apply quaternary scaling if in input
                    if bool(quaternary_energy_scaling):
                        if pair in quaternary_pairs_to_chain:
                            chain_ndx_tuple = quaternary_pairs_to_chain[pair]
                            HB_energy = HB_energy*quaternary_energy_scaling[chain_ndx_tuple]

                    break
                else:
                    HB_energy = np.array(dist_energy)[0,1]*energy_scaling
                    
                    #Apply quaternary scaling if in input
                    if bool(quaternary_energy_scaling):
                        if pair in quaternary_pairs_to_chain:
                            chain_ndx_tuple = quaternary_pairs_to_chain[pair]
                            HB_energy = HB_energy*quaternary_energy_scaling[chain_ndx_tuple]
                    
        else:
            HB_energy = np.array(dist_energy)[0,1]*energy_scaling
                    
            #Apply quaternary scaling if in input
            if bool(quaternary_energy_scaling):
                if pair in quaternary_pairs_to_chain:
                    chain_ndx_tuple = quaternary_pairs_to_chain[pair]
                    HB_energy = HB_energy*quaternary_energy_scaling[chain_ndx_tuple]

        mean_dist = np.mean(dists)  #Naive combination of minimas - ideally we would change the functional form, however we are very limited in GROMACS since tabulated potentials are currently unsupported
        #In practice this works well because the distance cutoff is small, so the standard deviation of the above mean is very small (i.e. the difference in distance between beads in two conformations having the same contact)
        
        harm_pairs.append([str(pair[0]+1),str(pair[1]+1),f"{harm_function_type}","{:.10f}".format(mean_dist),"{:.10f}".format(harm_k)])
        
        if networktype == "secondary":
            LJ_pairs.append([str(pair[0]+1),str(pair[1]+1),f"{LJ_function_type}","{:.10f}".format(mean_dist/np.power(2,1/6)),"{:.10f}".format(HB_energy+4.06)]) #Here we add back in the (unscaled) P2-P2 LJ energy that was exlcuded, to be consistent with the tertiary network
            excluded_pairs.append([str(pair[0]+1),str(pair[1]+1)])
        
        if networktype == "tertiary":
            LJ_pairs.append([str(pair[0]+1),str(pair[1]+1),f"{LJ_function_type}","{:.10f}".format(mean_dist/np.power(2,1/6)),"{:.10f}".format(HB_energy)])
    
    return harm_pairs,LJ_pairs,excluded_pairs,quaternary_pair_info

##### RUN PROGRAM #####
print(" o o o o o o o o o ")

all_secondary_pairs = []
all_secondary_pairs_dists_energies = []
all_tertiary_pairs = []
all_tertiary_pairs_dists_energies = []
    
#Filters to change between bead ndx and beadname, resname, and res ndx
filters = [pdbs_CG[0].top.to_dataframe()[0]["name"],pdbs_CG[0].top.to_dataframe()[0]["resName"],pdbs_CG[0].top.to_dataframe()[0]["resSeq"],pdbs_CG[0].top.to_dataframe()[0]["chainID"]]
a_ndx_to_bead_name = filters[0]
a_ndx_to_res_name = filters[1]
a_ndx_to_res_ndx = filters[2]
a_ndx_to_chain_ndx = filters[3]
#Interate through all input conformations and store the pair information seperately
for p,pdb_CG in enumerate(pdbs_CG):
    print("Processing conformation: {}".format(input_conformations[p]))
    if not silent:
        print("Computing distances between all beads - grab a coffee...")
    #Find all possible bead pairs
    all_pairs = np.array([[i,j] for i in np.arange(pdb_CG.top.n_atoms) for j in np.arange(i+1,pdb_CG.top.n_atoms)])
    
    #compute distances between all pairs and keep pairs that are below the cutoff
    all_dists = md.compute_distances(pdb_CG,all_pairs)[0]

    # The following if statement improves runtime if the secondary and tertiary cutoffs are the same
    if secondary_cutoff == tertiary_cutoff:
        cutoff_filter = all_dists < secondary_cutoff
        cut_pairs = all_pairs[cutoff_filter,:]
        cut_dists = all_dists[cutoff_filter]
    #Apply rules of the OLIVES model and split into secondary and tertiary networks, and find quaternary pair chain id
        cut_secondary_pairs,cut_secondary_dists,cut_tertiary_pairs,cut_tertiary_dists = knowledge_based_checks(cut_pairs,cut_dists,filters,silent)
    else:
        secondary_pairs,secondary_dists,tertiary_pairs,tertiary_dists = knowledge_based_checks(all_pairs,all_dists,filters,silent)
    
        #Apply secondary distance cutoff
        secondary_cutoff_filter = secondary_dists < secondary_cutoff
        cut_secondary_pairs = secondary_pairs[secondary_cutoff_filter,:]
        cut_secondary_dists = secondary_dists[secondary_cutoff_filter]
        
        #Apply tertiary distance cutoff
        tertiary_cutoff_filter = tertiary_dists < tertiary_cutoff
        cut_tertiary_pairs = tertiary_pairs[tertiary_cutoff_filter,:]
        cut_tertiary_dists = tertiary_dists[tertiary_cutoff_filter]
    
    #check secondary for HB pairs
    if not silent:
        print("Checking secondary network...")
    checked_secondary_pairs,checked_secondary_pairs_dists_energies = check_HB_pairs(cut_secondary_pairs,cut_secondary_dists,hbond_dict,energy_dict,filters,silent)
    
    #check tertiary for HB pairs
    if not silent:
        print("Checking tertiary network...")
    checked_tertiary_pairs,checked_tertiary_pairs_dists_energies = check_HB_pairs(cut_tertiary_pairs,cut_tertiary_dists,hbond_dict,energy_dict,filters,silent)
    
    #Secondary network maximum weight matching
    #We now impose that the secondary network can only have two HBs for each bead by performing two passes through a maximum weight matching    
    if not silent:
        print("Computing maximum weight matching of secondary network...")
    weighted_secondary_pairs_pass1 = [(pair[0][0],pair[0][1],pair[2]/pair[1]) for pair in checked_secondary_pairs_dists_energies]  #each pair is weighted by energy divided by distance
    secondary_graph_pass1 = nx.Graph()
    secondary_graph_pass1.add_weighted_edges_from(weighted_secondary_pairs_pass1)
    matched_secondary_network_pass1 = nx.max_weight_matching(secondary_graph_pass1,maxcardinality=False)  #maxcardinality=True leads to larger pair mismatch between similar structures because the pairs shift to fulfill maximum matching
    matched_secondary_pairs_pass1 = [tuple(sorted(pair)) for pair in matched_secondary_network_pass1]
    flat_matched_secondary_pairs_pass1 = [item for t in matched_secondary_pairs_pass1 for item in t]

    remaining_pairs1 = list(set(checked_secondary_pairs) - set(matched_secondary_pairs_pass1))
    
    weighted_secondary_pairs_pass2 = [(pair[0][0],pair[0][1],pair[2]/pair[1]) for pair in checked_secondary_pairs_dists_energies if pair[0] in remaining_pairs1]  #each pair is weighted by energy divided by distance
    secondary_graph_pass2 = nx.Graph()
    secondary_graph_pass2.add_weighted_edges_from(weighted_secondary_pairs_pass2)
    matched_secondary_network_pass2 = nx.max_weight_matching(secondary_graph_pass2,maxcardinality=False)
    matched_secondary_pairs_pass2 = [tuple(sorted(pair)) for pair in matched_secondary_network_pass2]
    flat_matched_secondary_pairs_pass2 = [item for t in matched_secondary_pairs_pass2 for item in t]

    #A third round of weighted pair matching is needed because subgraphs with an odd number of verticies cannot be fully paired, leaving gaps e.g. inside helices, this round fixes that
    remaining_pairs2 = []
    for pair in list(set(checked_secondary_pairs) - set(matched_secondary_pairs_pass1) - set(matched_secondary_pairs_pass2)):
        if pair[0] in flat_matched_secondary_pairs_pass1 and pair[0] in flat_matched_secondary_pairs_pass2:
            continue
        if pair[1] in flat_matched_secondary_pairs_pass1 and pair[1] in flat_matched_secondary_pairs_pass2:
            continue
        remaining_pairs2.append(pair)
    
    weighted_secondary_pairs_pass3 = [(pair[0][0],pair[0][1],pair[2]/pair[1]) for pair in checked_secondary_pairs_dists_energies if pair[0] in remaining_pairs2]  #each pair is weighted by energy divided by distance
    secondary_graph_pass3 = nx.Graph()
    secondary_graph_pass3.add_weighted_edges_from(weighted_secondary_pairs_pass3)
    matched_secondary_network_pass3 = nx.max_weight_matching(secondary_graph_pass3,maxcardinality=False)
    matched_secondary_pairs_pass3 = [tuple(sorted(pair)) for pair in matched_secondary_network_pass3]

    matched_secondary_pairs = sorted(matched_secondary_pairs_pass1 + matched_secondary_pairs_pass2 + matched_secondary_pairs_pass3)
    matched_secondary_pairs_dists_energies = [[pair[0],pair[1],pair[2]] for pair in checked_secondary_pairs_dists_energies if pair[0] in matched_secondary_pairs]

    #Tertinary network maximum weight matching
    if not silent:
        print("Computing maximum weight matching of tertiary network...")
    #Now we impose for the tertiary network that each bead can only have one hydrogen bond partner by maximum weight mathcing 
    #ASN, GLN, and ARG sidechains are included in a second round to allow them to form 2 HB
    weighted_tertiary_pairs_pass1 = [(pair[0][0],pair[0][1],(pair[2]/pair[1])) for pair in checked_tertiary_pairs_dists_energies]
    tertiary_graph_pass1 = nx.Graph()
    tertiary_graph_pass1.add_weighted_edges_from(weighted_tertiary_pairs_pass1)
    matched_tertiary_network_pass1 = nx.max_weight_matching(tertiary_graph_pass1,maxcardinality=False)
    matched_tertiary_pairs_pass1 = [tuple(sorted(pair)) for pair in matched_tertiary_network_pass1]
    flat_matched_tertiary_pairs_pass1 = [item for t in matched_tertiary_pairs_pass1 for item in t]

    remaining_pairs = list(set(checked_tertiary_pairs) - set(matched_tertiary_pairs_pass1))
  
    matched_teriary_pairs_pass1_residues = [(a_ndx_to_res_ndx[p[0]],a_ndx_to_res_ndx[p[1]]) for p in matched_tertiary_pairs_pass1]
    asn_gln_arg_pairs = []
    
    #TODO: The following decision tree is ugly but difficult to implement specific chemical information without it
    for p in remaining_pairs:
        if a_ndx_to_res_name[p[0]] == "GLN" or a_ndx_to_res_name[p[0]] == "ASN" or a_ndx_to_res_name[p[0]] == "ARG":
            if a_ndx_to_bead_name[p[0]] == 'BB':
                continue #Only consider the side chain bead of gln, asn, arg
            elif p[1] in flat_matched_tertiary_pairs_pass1:
                if a_ndx_to_bead_name[p[1]] == 'BB':
                    continue #if the partner is a BB and was paired in the first tertiary round, dont pair it again 
                elif a_ndx_to_res_name[p[1]] == "GLN" or a_ndx_to_res_name[p[1]] == "ASN" or a_ndx_to_res_name[p[1]] == "ARG":
                    asn_gln_arg_pairs.append(p) #pair with gln, asn, arg partnerts
                else:
                    continue 
            elif a_ndx_to_bead_name[p[1]] == 'BB':
                if (a_ndx_to_res_ndx[p[0]],a_ndx_to_res_ndx[p[1]]) in matched_teriary_pairs_pass1_residues:
                    continue #dont pair to the BB if already paired to the side chain
                else:
                    asn_gln_arg_pairs.append(p)
            else:
                asn_gln_arg_pairs.append(p)
        
        if a_ndx_to_res_name[p[1]] == "GLN" or a_ndx_to_res_name[p[1]] == "ASN" or a_ndx_to_res_name[p[1]] == "ARG":
            if a_ndx_to_bead_name[p[1]] == 'BB':
                continue #Only consider the side chain
            elif p[0] in flat_matched_tertiary_pairs_pass1:
                continue #if partner is already paired, only pair to gln,asn,arg residues which was taken care of above
            elif a_ndx_to_bead_name[p[0]] == 'BB':
                if (a_ndx_to_res_ndx[p[0]],a_ndx_to_res_ndx[p[1]]) in matched_teriary_pairs_pass1_residues:
                    continue #dont pair to the BB if already paired to the side chain
                else:
                    asn_gln_arg_pairs.append(p)
            else:
                asn_gln_arg_pairs.append(p)

    weighted_tertiary_pairs_pass2 = [(pair[0][0],pair[0][1],(pair[2]/pair[1])) for pair in checked_tertiary_pairs_dists_energies if pair[0] in asn_gln_arg_pairs]
    tertiary_graph_pass2 = nx.Graph()
    tertiary_graph_pass2.add_weighted_edges_from(weighted_tertiary_pairs_pass2)
    matched_tertiary_network_pass2 = nx.max_weight_matching(tertiary_graph_pass2,maxcardinality=False)
    matched_tertiary_pairs_pass2 = [tuple(sorted(pair)) for pair in matched_tertiary_network_pass2]
    
    matched_tertiary_pairs = sorted(matched_tertiary_pairs_pass1 + matched_tertiary_pairs_pass2)
    matched_tertiary_pairs_dists_energies = [[pair[0],pair[1],pair[2]] for pair in checked_tertiary_pairs_dists_energies if pair[0] in matched_tertiary_pairs]

    #collect data from each conformation
    all_secondary_pairs.append(matched_secondary_pairs)  
    all_secondary_pairs_dists_energies.append(matched_secondary_pairs_dists_energies)
    all_tertiary_pairs.append(matched_tertiary_pairs)  
    all_tertiary_pairs_dists_energies.append(matched_tertiary_pairs_dists_energies)
    
    if not silent:
        print("All possible pairs: {}".format(all_pairs.shape[0]))
        print("Secondary pairs after distance cutoff: {}".format(cut_secondary_pairs.shape[0]))
        print("Tertiary pairs after distance cutoff: {}".format(cut_tertiary_pairs.shape[0]))
        print("Secondary HB pairs after checks: {}".format(len(checked_secondary_pairs)))
        print("Tertiary HB pairs after checks: {}".format(len(checked_tertiary_pairs)))
        print("Secondary HB pairs after weighted matching: {}".format(len(matched_secondary_pairs)))
        print("Tertiary HB pairs after weighted matching: {}".format(len(matched_tertiary_pairs)))
        print(" o o o o o o o o o ")



info_secondary_pairs, info_tertiary_pairs, unique_secondary_pairs_for_scaling, unique_tertiary_pairs_for_scaling = get_pair_set_information(input_conformations,all_secondary_pairs,all_tertiary_pairs)

#For each pair, check for if the pair occurs multiple times among all conformations 
secondary_pair_multiples_dict = construct_pair_multiples_dict(all_secondary_pairs,all_secondary_pairs_dists_energies)
tertiary_pair_multiples_dict = construct_pair_multiples_dict(all_tertiary_pairs,all_tertiary_pairs_dists_energies)

#Combine minimas and format the output
secondary_harm_potentials,secondary_LJ_potentials,secondary_exclusions,secondary_quaternary_pair_info = combine_and_format_potentials(secondary_pair_multiples_dict,"secondary",unique_secondary_pairs_for_scaling,unique_pair_scaling,secondary_energy_scaling,quaternary_energy_scaling,filters)
tertiary_harm_potentials,tertiary_LJ_potentials,tertiary_exclusions,tertiary_quaternary_pair_info = combine_and_format_potentials(tertiary_pair_multiples_dict,"tertiary",unique_tertiary_pairs_for_scaling,unique_pair_scaling,tertiary_energy_scaling,quaternary_energy_scaling,filters)

##### WRAP-UP #####
if not silent:
    print("Wrapping up...")
if args.extend_itp: 
    #Write the OLIVES model into the protein topology"
    with open(itp_CG, "a") as f:
        output = ['; OLIVES exclusions for secondary LJ 1-4 pairs'.split()] + ['[ exclusions ]'.split()] + secondary_exclusions + [[""]]
        for line in output:
            f.write("{}\n".format(' '.join(line)))
        output = ['; OLIVES secondary as LJ 1-4 pairs'.split()] + ['[ pairs ]'.split()] + secondary_LJ_potentials + [[""]]
        for line in output:
            f.write("{}\n".format(' '.join(line)))
        output = ['; OLIVES tertiary as LJ 1-4 pairs'.split()] + tertiary_LJ_potentials + [[""]]
        for line in output:
            f.write("{}\n".format(' '.join(line)))

if args.write_separate_itp:
    print("If you use the --write_separate_itp True and --extend_itp False. Remember to include OLIVES itp's in your topology right after {}".format(itp_CG))
    print("Exclusion .itps should be above the interaction .itps")

    #Write the model as harmonic elastic network
    with open("Harm_secondary_"+itp_CG,'w') as mo:
        output = [[""]] + ['; OLIVES secondary as harmonic bonds'.split()] + ['[ bonds ]'.split()] + secondary_harm_potentials + [[""]]
        for line in output:
            mo.write("{}\n".format(' '.join(line)))
    print("Wrote: {}".format("Harm_secondary_"+itp_CG))
    
    with open("Harm_tertiary_"+itp_CG,'w') as mo:
        output = [[""]] + ['; OLIVES tertiary as harmonic bonds'.split()] + ['[ bonds ]'.split()] + tertiary_harm_potentials + [[""]]
        for line in output:
            mo.write("{}\n".format(' '.join(line)))
    print("Wrote: {}".format("Harm_tertiary_"+itp_CG))

    #Write the HB go-bonds
    with open("LJ_secondary_"+itp_CG,'w') as mo:
        output = [[""]] + ['; OLIVES secondary as LJ 1-4 pairs'.split()] + ['[ pairs ]'.split()] + secondary_LJ_potentials + [[""]]
        for line in output:
            mo.write("{}\n".format(' '.join(line)))
    print("Wrote: {}".format("LJ_secondary_"+itp_CG))
    
    with open("LJ_tertiary_"+itp_CG,'w') as mo:
        output = [[""]] + ['; OLIVES tertiary as LJ 1-4 pairs'.split()] + ['[ pairs ]'.split()] + tertiary_LJ_potentials + [[""]]
        for line in output:
            mo.write("{}\n".format(' '.join(line)))
    print("Wrote: {}".format("LJ_tertiary_"+itp_CG))
    
    #Write exlucsion file for the secondary network
    with open("Exclusions_secondary_"+itp_CG,'w') as mo:
        output = [[""]] + ['; OLIVES exclusions for secondary LJ 1-4 pairs'.split()] + ['[ exclusions ]'.split()] + secondary_exclusions + [[""]]
        for line in output:
            mo.write("{}\n".format(' '.join(line)))
    print("Wrote: {}".format("Exclusions_secondary_"+itp_CG))

if args.write_vmd_itp: 
    print('"OLIVES_VMD_".itp can be used for VMD visualization of the secondary and tertiary networks with cg_bonds-v5.tcl: 1: Load your CG pdb, 2: source cg_bonds-v5.tcl 3: cg_bonds -cutoff 100 -top "SS.top" (you have to create this dummy top yourself and include the OLIVES_VMD_ itp) 4: Representation -> bonds 5: repeat for tertiary network 6: profit')
    #Write visualization itp for VMD "cg_bonds"
    with open(itp_CG, "r") as f:
        line_cols = []
        for line in f:
            cols = line.split()
            if any(s=='bonds' for s in cols):
                break
            line_cols.append(cols)
    
    with open("OLIVES_VMD_secondary_"+itp_CG,'w') as o:
        for line in line_cols:
            o.write("{}\n".format(' '.join(line)))
        output = [[""]] + ['; OLIVES secondary as harmonic bonds'.split()] + ['[ bonds ]'.split()] + secondary_harm_potentials + [[""]]
        for line in output:
            o.write("{}\n".format(' '.join(line)))
    print("Wrote: {}".format("OLIVES_VMD_secondary_"+itp_CG))
    
    with open("OLIVES_VMD_tertiary_"+itp_CG,'w') as o:
        for line in line_cols:
            o.write("{}\n".format(' '.join(line)))
        output = [[""]] + ['; OLIVES tertiary as harmonic bonds'.split()] + ['[ bonds ]'.split()] + tertiary_harm_potentials + [[""]]
        for line in output:
            o.write("{}\n".format(' '.join(line)))
    print("Wrote: {}".format("OLIVES_VMD_tertiary_"+itp_CG))

if args.write_bond_information_file:
    itp_name = itp_CG.split('.itp')[0]
    with open("OLIVES_info_secondary_pairs_"+itp_name+".dat",'w') as mo:
        for line in info_secondary_pairs:
            mo.write("{}\n".format(' '.join(line)))
    print("Wrote: {}".format("OLIVES_info_secondary_pairs_"+itp_name+".dat"))
            
    with open("OLIVES_info_tertiary_pairs_"+itp_name+".dat",'w') as mo:
        for line in info_tertiary_pairs:
            mo.write("{}\n".format(' '.join(line)))
    print("Wrote: {}".format("OLIVES_info_tertiary_pairs_"+itp_name+".dat"))

    if bool(quaternary_energy_scaling):
        with open("OLIVES_info_quaternary_pairs_"+itp_name+".dat",'w') as mo:
            output = ['; OLIVES secondary pairs in quaternary bonds'.split()] + ['; pair chainID'.split()] + secondary_quaternary_pair_info
            for line in output:
                mo.write("{}\n".format(' '.join(line)))
            output = ['; OLIVES tertiary pairs in quaternary bonds'.split()] + ['; pair chainID'.split()] + tertiary_quaternary_pair_info
            for line in output:
                mo.write("{}\n".format(' '.join(line)))
        print("Wrote: {}".format("OLIVES_info_quaternary_pairs_"+itp_name+".dat"))


if not silent:
    print('Remember to use "gmx mdrun --noddcheck -rdd 2.0" to avoid domain decomposition warnings if OLIVES bonds gets too long.')
    print('"-noddcheck" turns off a GROMACS domain decomposition error for pairs that become too long relative to the length of a domain (set by -rdd), abruptly ending your run.')
    print('This could happen if a protein complex dissociates or a protein unfolds. The 2 nm cutoff for domains in -rdd is where a LJ potential with energy minimum distance at 0.55 nm (the OLIVES cutoff) goes to 0, and therefore not important if missed.')
    print("If you experience DD errors that cannot be solved by --noddcheck, try using 1 MPI tread and all openMP threads instead of domain decomposition https://manual.gromacs.org/documentation/current/user-guide/mdrun-performance.html.")
    print("OLIVES has been tested using the -scfix flag of martinize2 - if you are using multistate OLIVES, remember to combine the -scfix flag with the OLIVES scfix script")
    print("OLIVES does not use the virtual site approach implemented in e.g. GoMARTINI and thus does not use the martinize2 flags -govs-include.")
    print("OLIVES does not require -dssp secondary structure restraints, although one can opt to restrain the secondary structure at the cost of overstabilized secondary structure.")

print(" o o o o o o o o o ")
print("Done!")

