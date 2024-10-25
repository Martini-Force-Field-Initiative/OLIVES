#!/usr/bin/env python
print("OLIVES_v2.0.1_M3.0.0.py is written by Kasper Busk Pedersen, 29th of July 2024.")
print("If you use this script in your work, please cite Pedersen et al. https://pubs.acs.org/doi/10.1021/acs.jctc.4c00553 DOI: 10.1021/acs.jctc.4c00553")

### IMPORT ###
import math
import argparse
import numpy as np
import mdtraj as md
import networkx as nx

### DEFINITION OF THE MODEL ###
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

#Residues allowed to form 2 hydrogen bonds for the side chain bead i.e. enter second matching round in the tertiary network (still adheres to HBA-HBD rules) 
special_residues = {"GLN", "ASN", "ARG"} #HIS not included since it is mapped with higher resolution, allowing multiple HBs per side chain in the first matching round 

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


##### INPUT FUNCTIONS #####
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
    return parser.parse_args()

def check_topologies(pdbs_CG):
    #Check of the topology of the input conformations match
    natoms = [pdb.n_atoms for pdb in pdbs_CG]
    nresidues = [pdb.n_residues for pdb in pdbs_CG]
    nchains = [pdb.n_chains for pdb in pdbs_CG]
    if len(set(natoms)) > 1:
        raise ValueError('Input conformations do not have the same number of beads. Please check your topology.')
    if len(set(nresidues)) > 1:
        raise ValueError('Input conformations do not have the same number of residues. Please check your topology.')
    if len(set(nchains)) > 1:
        raise ValueError('Input conformations do not have the same number of chains. Please check your topology.')

def verify_scaling_inputs(quaternary_energy_scaling, nchains, input_conformations, unique_pair_scaling):
    #If quaternary structure scaling is provided check that the dictionary is the correct size (does not correct user errors in tuples)
    if bool(quaternary_energy_scaling):
        expected_len = int(math.factorial(nchains) / (2 * math.factorial(nchains - 2)))
        if len(quaternary_energy_scaling) != expected_len:
            raise ValueError('Incorrect number of quaternary structure scaling values. Expected {}.'.format(expected_len))
    
    #Check if the number of conformations match in the number of scaling values
    if not len(input_conformations) == len(unique_pair_scaling):
        raise ValueError('The number of input conformations do not match the number of scaling values provided by the --unique_pair_scaling flag.')

##### HELPER FUNCTIONS #####

def detect_cys_bridges(cut_pairs, cut_dists, a_ndx_to_res_name, a_ndx_to_bead_name):
    # Detects disulfide bridges between cysteine residues within a cutoff distance.
    # Returns a list of detected disulfide bridges.
    cys_bridges = [
        pair for i, pair in enumerate(cut_pairs)
        if a_ndx_to_res_name[pair[0]] == 'CYS' and a_ndx_to_res_name[pair[1]] == 'CYS' and
        (a_ndx_to_bead_name[pair[0]] != 'BB' and a_ndx_to_bead_name[pair[1]] != 'BB') and
        cut_dists[i] < 0.3
    ]
    print(f"Detected {len(cys_bridges)} disulfide bridges")
    return cys_bridges

def should_ignore_cys(pair, cys_bridges, a_ndx_to_res_ndx):
    # Determines if a cysteine pair should be ignored based on disulfide bridge presence.
    # Returns True if the pair should be ignored, otherwise False.
    for b in cys_bridges:
        if np.array_equal(pair, b):
            return True
        if a_ndx_to_res_ndx[pair[0]] == a_ndx_to_res_ndx[b[0]] and a_ndx_to_res_ndx[pair[1]] == a_ndx_to_res_ndx[b[1]]:
            return True
        if a_ndx_to_res_ndx[pair[0]] == a_ndx_to_res_ndx[b[1]] and a_ndx_to_res_ndx[pair[1]] == a_ndx_to_res_ndx[b[0]]:
            return True
    return False

def process_pairs(cut_pairs, cut_dists, filters, cys_bridges):
    # Processes pairs of beads to categorize them into secondary and tertiary interactions.
    # Applies rules to filter out invalid pairs and considers cysteine bridges.
    a_ndx_to_bead_name, a_ndx_to_res_name, a_ndx_to_res_ndx = filters[:3]
    pairs_secondary, dists_secondary = [], []
    pairs_tertiary, dists_tertiary = [], []

    for i, pair in enumerate(cut_pairs):
        if a_ndx_to_bead_name[pair[0]] == 'CA' or a_ndx_to_bead_name[pair[1]] == 'CA':
            raise ValueError('Atomistic structure detected or unexpected -govs-includes in martinize2.')
        if a_ndx_to_res_ndx[pair[0]] == a_ndx_to_res_ndx[pair[1]]:
            continue #exclude intra-residue hbonds
        
        #Rules for the secondary network
        if a_ndx_to_bead_name[pair[0]] == 'BB' and a_ndx_to_bead_name[pair[1]] == 'BB':
            if a_ndx_to_res_ndx[pair[0]] + 1 == a_ndx_to_res_ndx[pair[1]]:
                continue
            if a_ndx_to_res_ndx[pair[0]] + 2 == a_ndx_to_res_ndx[pair[1]]:
                continue
            else:
                pairs_secondary.append(pair)
                dists_secondary.append(cut_dists[i])
                continue
        
        #Rules for tertiary network
        if a_ndx_to_bead_name[pair[0]] == 'BB' or a_ndx_to_bead_name[pair[1]] == 'BB':
            if a_ndx_to_res_ndx[pair[0]]+1 == (a_ndx_to_res_ndx[pair[1]]):
                continue # exclude BB <-> sidechain+1 hbonds

        if not should_ignore_cys(pair, cys_bridges, a_ndx_to_res_ndx):
            pairs_tertiary.append(pair)
            dists_tertiary.append(cut_dists[i])

    return np.array(pairs_secondary), np.array(dists_secondary), np.array(pairs_tertiary), np.array(dists_tertiary)

def apply_cutoff(pairs, dists, cutoff):
    # Applies a distance cutoff to filter pairs of beads.
    # Returns pairs and distances that are within the cutoff distance.
    cutoff_filter = dists < cutoff
    cut_pairs = pairs[cutoff_filter]
    cut_dists = dists[cutoff_filter]
    return cut_pairs, cut_dists

def create_filters(pdbs_CG):
    # Creates filters for bead names, residue names, residue indices, and chain indices.
    # Returns a list of filters for use in other functions.
    dataframe = pdbs_CG.top.to_dataframe()[0]
    filters = [
        dataframe["name"],     # Bead names
        dataframe["resName"],  # Residue names
        dataframe["resSeq"],   # Residue indices
        dataframe["chainID"]   # Chain indices
    ]
    return filters

def get_hbond_info(pair, filters, hbond_dict):
    # Retrieves hydrogen bond information for a given pair of beads.
    # Returns bead names, residue names, and hydrogen bond lists.
    a_ndx_to_bead_name, a_ndx_to_res_name = filters[0], filters[1]
    bead_names = [a_ndx_to_bead_name[idx] for idx in pair]
    res_names = [a_ndx_to_res_name[idx] for idx in pair]
    hbond_lists = [hbond_dict[res_names[i]][bead_names[i]] for i in range(2)]
    return bead_names, res_names, hbond_lists

def calculate_hb_energy(hbond_lists, energy_dict):
    # Calculates the hydrogen bond energy between two beads.
    # Returns the maximum hydrogen bond energy found.
    HBA_HBD_partners = hbond_lists[0][1][0] + "-" + hbond_lists[1][1][1]
    HBD_HBA_partners = hbond_lists[1][1][0] + "-" + hbond_lists[0][1][1]
    HBA_HBD_energy = energy_dict.get(HBA_HBD_partners, 0)
    HBD_HBA_energy = energy_dict.get(HBD_HBA_partners, 0)
    return max(HBA_HBD_energy, HBD_HBA_energy)

def check_hb_pair(pair, dists, filters, hbond_dict, energy_dict, index):
    # Checks if a pair of beads forms a hydrogen bond and calculates its energy.
    # Returns the pair, distance, and hydrogen bond energy if valid, otherwise None.
    bead_names, res_names, hbond_lists = get_hbond_info(pair, filters, hbond_dict)
    if hbond_lists[0][0][0] and hbond_lists[1][0][1]:  # First bead is HBA and second is HBD
        if hbond_lists[0][0][1] and hbond_lists[1][0][0]:  # First bead is also HBD and second is also HBA
            HB_energy = calculate_hb_energy(hbond_lists, energy_dict)
            return [(pair[0], pair[1]), dists[index], HB_energy]

    if hbond_lists[0][0][0] and hbond_lists[1][0][1]:  # First bead is HBA and second is HBD
        HB_partners = hbond_lists[0][1][0] + "-" + hbond_lists[1][1][1]
        HB_energy = energy_dict.get(HB_partners, 0)
        return [(pair[0], pair[1]), dists[index], HB_energy]

    if hbond_lists[0][0][1] and hbond_lists[1][0][0]:  # First bead is HBD and second is HBA
        HB_partners = hbond_lists[1][1][0] + "-" + hbond_lists[0][1][1]
        HB_energy = energy_dict.get(HB_partners, 0)
        return [(pair[0], pair[1]), dists[index], HB_energy]

    return None

def create_weighted_pairs(pairs_dists_energies):
    # Creates weighted pairs for graph-based matching from pair distances and energies.
    return [(pair[0][0], pair[0][1], pair[2] / pair[1]) for pair in pairs_dists_energies]

def perform_max_weight_matching(weighted_pairs):
    # Performs maximum weight matching on the given pairs.
    graph = nx.Graph()
    graph.add_weighted_edges_from(weighted_pairs)
    matched_network = nx.max_weight_matching(graph, maxcardinality=False)
    return [tuple(sorted(pair)) for pair in matched_network]

def flatten_matched_pairs(matched_pairs):
    return [item for pair in matched_pairs for item in pair]

def get_remaining_pairs(checked_pairs, matched_pairs1, matched_pairs2=None):
    # Gets pairs that are not matched in the first or second matching pass.
    remaining_pairs = set(checked_pairs) - set(matched_pairs1)
    if matched_pairs2:
        remaining_pairs -= set(matched_pairs2)
    return list(remaining_pairs)

def get_special_case_pairs(remaining_pairs, flat_matched_pairs, a_ndx_to_res_name, a_ndx_to_bead_name, special_residues):
    # Identifies special case pairs involving specified residues.
    # Returns a list of special case pairs.
    special_case_pairs = []
    for p in remaining_pairs:
        if a_ndx_to_res_name[p[0]] in special_residues:
            if a_ndx_to_bead_name[p[0]] == 'BB':
                continue
            if p[1] in flat_matched_pairs:
                if a_ndx_to_bead_name[p[1]] == 'BB':
                    continue
                if a_ndx_to_res_name[p[1]] in special_residues:
                    special_case_pairs.append(p)
                else:
                    continue
            else:
                special_case_pairs.append(p)

        if a_ndx_to_res_name[p[1]] in special_residues:
            if a_ndx_to_bead_name[p[1]] == 'BB':
                continue
            if p[0] not in flat_matched_pairs:
                special_case_pairs.append(p)

    return special_case_pairs

def get_pair_set_information(input_conformations, all_secondary_pairs, all_tertiary_pairs):
    # Collects information about pairs across all conformations.
    # Returns detailed pair information across conformations including unique pairs and intersections of pair sets.
    def add_pairs_info(info_list, conformation, pairs):
        info_list.append(f'; pairs in {conformation}'.split())
        info_list.append(str(sorted([(p[0] + 1, p[1] + 1) for p in pairs])).split())

    def add_unique_pairs_info(info_list, conformation, unique_pairs, unique_pairs_for_scaling):
        info_list.append(f'; pairs unique to {conformation}'.split())
        info_list.append(str(sorted([(p[0] + 1, p[1] + 1) for p in unique_pairs])).split())
        unique_pairs_for_scaling.append(sorted([(p[0], p[1]) for p in unique_pairs]))

    def add_intersection_info(info_list, conformations, intersection):
        info_list.append(f'; intersection of {str(conformations)}'.split())
        info_list.append(str(sorted([(p[0] + 1, p[1] + 1) for p in intersection])).split())

    info_secondary_pairs = []
    info_tertiary_pairs = []
    unique_secondary_pairs_for_scaling = []
    unique_tertiary_pairs_for_scaling = []

    if len(input_conformations) == 1:
        add_pairs_info(info_secondary_pairs, input_conformations[0], all_secondary_pairs[0])
        add_pairs_info(info_tertiary_pairs, input_conformations[0], all_tertiary_pairs[0])
    elif len(input_conformations) > 1:
        conf_ndx = range(len(input_conformations))
        for c in conf_ndx:
            add_pairs_info(info_secondary_pairs, input_conformations[c], all_secondary_pairs[c])
            add_pairs_info(info_tertiary_pairs, input_conformations[c], all_tertiary_pairs[c])

        for c in conf_ndx:
            conf_set_secondary = set(all_secondary_pairs[c])
            other_confs_secondary = [all_secondary_pairs[i] for i in conf_ndx if i != c]
            other_confs_set_secondary = set(item for sublist in other_confs_secondary for item in sublist)
            unique_pairs_secondary = conf_set_secondary - other_confs_set_secondary
            add_unique_pairs_info(info_secondary_pairs, input_conformations[c], unique_pairs_secondary, unique_secondary_pairs_for_scaling)

            conf_set_tertiary = set(all_tertiary_pairs[c])
            other_confs_tertiary = [all_tertiary_pairs[i] for i in conf_ndx if i != c]
            other_confs_set_tertiary = set(item for sublist in other_confs_tertiary for item in sublist)
            unique_pairs_tertiary = conf_set_tertiary - other_confs_set_tertiary
            add_unique_pairs_info(info_tertiary_pairs, input_conformations[c], unique_pairs_tertiary, unique_tertiary_pairs_for_scaling)

        secondary_intersection = set.intersection(*[set(l) for l in all_secondary_pairs])
        add_intersection_info(info_secondary_pairs, input_conformations, secondary_intersection)

        tertiary_intersection = set.intersection(*[set(l) for l in all_tertiary_pairs])
        add_intersection_info(info_tertiary_pairs, input_conformations, tertiary_intersection)

    return info_secondary_pairs, info_tertiary_pairs, unique_secondary_pairs_for_scaling, unique_tertiary_pairs_for_scaling

def construct_pair_multiples_dict(all_pairs,all_pairs_dists_energies):
    # Constructs a dictionary of pairs with multiple occurrences.
    # Used for averaging distances and energies across conformations.
    pairs_dict = {}
    for s,structure_pairs in enumerate(all_pairs):
        for p,pair in enumerate(structure_pairs):
            current_entries = pairs_dict.keys()
            if pair in current_entries:
                currect_entry = pairs_dict[pair]
                pairs_dict.update({pair: currect_entry+[[all_pairs_dists_energies[s][p][1],all_pairs_dists_energies[s][p][2]]]})
            else:
                pairs_dict.update({pair: [[all_pairs_dists_energies[s][p][1],all_pairs_dists_energies[s][p][2]]]})
    return pairs_dict

def write_output(file_name, output):
    # Writes the output to a specified file.
    with open(file_name, 'w') as f:
        for line in output:
            f.write("{}\n".format(' '.join(line)))
    print(f"Wrote: {file_name}")

def write_vmd(file_name, pre_itp_lines, output):
    # Writes the output to a specified file.
    with open(file_name, 'w') as f:
        for line in pre_itp_lines:
            f.write("{}\n".format(' '.join(line)))
        for line in output:
            f.write("{}\n".format(' '.join(line)))
    print(f"Wrote: {file_name}")

def append_to_itp(file_name, output):
    # Appends the output to an existing ITP file.
    with open(file_name, 'a') as f:
        for line in output:
            f.write("{}\n".format(' '.join(line)))

##### MAIN FUNCTIONS #####

def knowledge_based_checks(cut_pairs, cut_dists, filters):
    # Performs knowledge-based checks to detect and process cysteine bridges.
    # Returns categorized secondary and tertiary interaction pairs.
    a_ndx_to_bead_name, a_ndx_to_res_name, a_ndx_to_res_ndx = filters[:3]
    cys_bridges = detect_cys_bridges(cut_pairs, cut_dists, a_ndx_to_res_name, a_ndx_to_bead_name)
    
    # Processes pairs of beads to categorize them into secondary and tertiary interactions.
    # Applies rules to filter out invalid pairs and considers cysteine bridges.
    pairs_secondary, dists_secondary = [], []
    pairs_tertiary, dists_tertiary = [], []

    for i, pair in enumerate(cut_pairs):
        if a_ndx_to_bead_name[pair[0]] == 'CA' or a_ndx_to_bead_name[pair[1]] == 'CA':
            raise ValueError('Atomistic structure detected or unexpected -govs-includes in martinize2.')
        if a_ndx_to_res_ndx[pair[0]] == a_ndx_to_res_ndx[pair[1]]:
            continue #exclude intra-residue hbonds
        
        #Rules for the secondary network
        if a_ndx_to_bead_name[pair[0]] == 'BB' and a_ndx_to_bead_name[pair[1]] == 'BB':
            if a_ndx_to_res_ndx[pair[0]] + 1 == a_ndx_to_res_ndx[pair[1]]:
                continue
            if a_ndx_to_res_ndx[pair[0]] + 2 == a_ndx_to_res_ndx[pair[1]]:
                continue
            else:
                pairs_secondary.append(pair)
                dists_secondary.append(cut_dists[i])
                continue
        
        #Rules for tertiary network
        if a_ndx_to_bead_name[pair[0]] == 'BB' or a_ndx_to_bead_name[pair[1]] == 'BB':
            if a_ndx_to_res_ndx[pair[0]]+1 == (a_ndx_to_res_ndx[pair[1]]):
                continue # exclude BB <-> sidechain+1 hbonds

        if not should_ignore_cys(pair, cys_bridges, a_ndx_to_res_ndx):
            pairs_tertiary.append(pair)
            dists_tertiary.append(cut_dists[i])

    return np.array(pairs_secondary), np.array(dists_secondary), np.array(pairs_tertiary), np.array(dists_tertiary)

def check_HB_pairs(pairs, dists, hbond_dict, energy_dict, filters):
    # Checks pairs for hydrogen bonding potential and calculates energies.
    # Returns valid hydrogen bond pairs and their energies.
    checked_pairs = []
    checked_pairs_dists_energies = []

    for i, pair in enumerate(pairs):
        result = check_hb_pair(pair, dists, filters, hbond_dict, energy_dict, i)
        if result:
            checked_pairs.append((pair[0], pair[1]))
            checked_pairs_dists_energies.append(result)

    return checked_pairs, checked_pairs_dists_energies

def match_secondary_network(checked_pairs_dists_energies, checked_pairs):
    # Matches pairs in the secondary network using maximum weight matching.
    # Returns matched secondary pairs and their distances and energies.
    weighted_pairs_pass1 = create_weighted_pairs(checked_pairs_dists_energies)
    matched_pairs_pass1 = perform_max_weight_matching(weighted_pairs_pass1)
    flat_matched_pairs_pass1 = flatten_matched_pairs(matched_pairs_pass1)

    remaining_pairs1 = list(set(checked_pairs) - set(matched_pairs_pass1))
    weighted_pairs_pass2 = create_weighted_pairs([pair for pair in checked_pairs_dists_energies if pair[0] in remaining_pairs1])
    matched_pairs_pass2 = perform_max_weight_matching(weighted_pairs_pass2)
    flat_matched_pairs_pass2 = flatten_matched_pairs(matched_pairs_pass2)

    remaining_pairs2 = []
    for pair in list(set(checked_pairs) - set(matched_pairs_pass1) - set(matched_pairs_pass2)):
        if pair[0] in flat_matched_pairs_pass1 and pair[0] in flat_matched_pairs_pass2:
            continue
        if pair[1] in flat_matched_pairs_pass1 and pair[1] in flat_matched_pairs_pass2:
            continue
        remaining_pairs2.append(pair)

    weighted_pairs_pass3 = create_weighted_pairs([pair for pair in checked_pairs_dists_energies if pair[0] in remaining_pairs2])
    matched_pairs_pass3 = perform_max_weight_matching(weighted_pairs_pass3)

    matched_secondary_pairs = sorted(matched_pairs_pass1 + matched_pairs_pass2 + matched_pairs_pass3)
    matched_secondary_pairs_dists_energies = [
        [pair[0], pair[1], pair[2]] for pair in checked_pairs_dists_energies if pair[0] in matched_secondary_pairs
    ]
    return matched_secondary_pairs, matched_secondary_pairs_dists_energies


def match_tertiary_network(checked_pairs_dists_energies, checked_pairs, a_ndx_to_res_name, a_ndx_to_bead_name, special_residues):
    # Matches pairs in the tertiary network using maximum weight matching.
    # Identifies special cases involving specified residues.
    weighted_pairs_pass1 = create_weighted_pairs(checked_pairs_dists_energies)
    matched_pairs_pass1 = perform_max_weight_matching(weighted_pairs_pass1)
    flat_matched_pairs_pass1 = flatten_matched_pairs(matched_pairs_pass1)

    remaining_pairs1 = get_remaining_pairs(checked_pairs, matched_pairs_pass1)
    special_case_pairs = get_special_case_pairs(remaining_pairs1, flat_matched_pairs_pass1, a_ndx_to_res_name, a_ndx_to_bead_name, special_residues)

    weighted_pairs_pass2 = create_weighted_pairs([pair for pair in checked_pairs_dists_energies if pair[0] in special_case_pairs])
    matched_pairs_pass2 = perform_max_weight_matching(weighted_pairs_pass2)

    matched_tertiary_pairs = sorted(matched_pairs_pass1 + matched_pairs_pass2)
    matched_tertiary_pairs_dists_energies = [
        [pair[0], pair[1], pair[2]] for pair in checked_pairs_dists_energies if pair[0] in matched_tertiary_pairs
    ]
    return matched_tertiary_pairs, matched_tertiary_pairs_dists_energies

def combine_and_format_potentials(pairs_dict,networktype,unique_pairs,unique_pair_scaling,energy_scaling,quaternary_energy_scaling,filters):
    # Formats interaction potentials for inclusion in GROMACS ITP files.
    # Combines distances and energies for multiple pairs.
    a_ndx_to_chain_ndx = filters[3]
    harm_function_type = 1
    LJ_function_type = 1
    harm_k = 700
    
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

def generate_output(args, itp_CG, secondary_exclusions, secondary_LJ_potentials, tertiary_LJ_potentials, 
                    secondary_harm_potentials, tertiary_harm_potentials, info_secondary_pairs, 
                    info_tertiary_pairs, secondary_quaternary_pair_info, tertiary_quaternary_pair_info, 
                    quaternary_energy_scaling):
    # Generates output files with the formatted potentials and pair information.
    # Writes the outputs to specified files or appends to existing ITP files.
    if args.extend_itp: 
        # Write the OLIVES model into the protein topology
        append_to_itp(itp_CG, ['; OLIVES exclusions for secondary LJ 1-4 pairs'.split()] + ['[ exclusions ]'.split()] + secondary_exclusions + [[""]])
        append_to_itp(itp_CG, ['; OLIVES secondary as LJ 1-4 pairs'.split()] + ['[ pairs ]'.split()] + secondary_LJ_potentials + [[""]])
        append_to_itp(itp_CG, ['; OLIVES tertiary as LJ 1-4 pairs'.split()] + tertiary_LJ_potentials + [[""]])

    if args.write_separate_itp:
        print(f"If you use the --write_separate_itp True and --extend_itp False. Remember to include OLIVES itp's in your topology right after {itp_CG}")
        print("Exclusion .itps should be above the interaction .itps")
        
        # Write the model as harmonic elastic network
        write_output(f"Exclusions_secondary_{itp_CG}", [[""]] + ['; OLIVES exclusions for secondary LJ 1-4 pairs'.split()] + ['[ pairs ]'.split()] + secondary_LJ_potentials + [[""]])
        write_output(f"LJ_secondary_{itp_CG}", [[""]] + ['; OLIVES secondary as LJ 1-4 pairs'.split()] + ['[ pairs ]'.split()] + secondary_LJ_potentials + [[""]])
        write_output(f"LJ_tertiary_{itp_CG}", [[""]] + ['; OLIVES tertiary as LJ 1-4 pairs'.split()] + ['[ pairs ]'.split()] + tertiary_LJ_potentials + [[""]])

        # Write the model as harmonic elastic network
        write_output(f"Harm_secondary_{itp_CG}", [[""]] + ['; OLIVES secondary as harmonic bonds'.split()] + ['[ bonds ]'.split()] + secondary_harm_potentials + [[""]])
        write_output(f"Harm_tertiary_{itp_CG}", [[""]] + ['; OLIVES tertiary as harmonic bonds'.split()] + ['[ bonds ]'.split()] + tertiary_harm_potentials + [[""]])

    if args.write_vmd_itp:
        #Write visualization itp for VMD "cg_bonds"
        with open(itp_CG, "r") as f:
            pre_itp_lines = []
            for line in f:
                cols = line.split()
                if any(s=='bonds' for s in cols):
                    break
                pre_itp_lines.append(cols)
        
        write_vmd(f"OLIVES_VMD_secondary_{itp_CG}",pre_itp_lines, [[""]] + ['; OLIVES secondary as harmonic bonds'.split()] + ['[ bonds ]'.split()] + secondary_harm_potentials + [[""]])
        write_vmd(f"OLIVES_VMD_tertiary_{itp_CG}",pre_itp_lines, [[""]] + ['; OLIVES tertiary as harmonic bonds'.split()] + ['[ bonds ]'.split()] + tertiary_harm_potentials + [[""]])
        

    if args.write_bond_information_file:
        write_output("OLIVES_info_secondary_pairs.dat", info_secondary_pairs)
        write_output("OLIVES_info_tertiary_pairs.dat", info_tertiary_pairs)

        if bool(quaternary_energy_scaling):
            write_output("OLIVES_info_quaternary_pairs.dat", ['; OLIVES secondary pairs in quaternary bonds'.split()] + ['; pair chainID'.split()] + secondary_quaternary_pair_info +
                                                         ['; OLIVES tertiary pairs in quaternary bonds'.split()] + ['; pair chainID'.split()] + tertiary_quaternary_pair_info)


### MAIN ###

def main():
    # Parses user input and loads protein conformations.
    # Processes interactions, matches networks, and generates output files.

    args = user_input()
    input_conformations = list(args.c.split(","))
    itp_CG = args.i
    pdbs_CG = [md.load(pdb) for pdb in input_conformations]
    secondary_cutoff = args.ss_cutoff  #[nm] - Distance cutoff for defining a hbond 
    tertiary_cutoff = args.ts_cutoff  #[nm] - Distance cutoff for defining a hbond  
    secondary_energy_scaling = args.ss_scaling  #[kJ/mol] - Is multiplied with the relative hbond energies 
    tertiary_energy_scaling = args.ts_scaling  #[kJ/mol] - Is multiplied with the relative hbond energies
    quaternary_energy_scaling = eval(args.qs_scaling)
    unique_pair_scaling = [float(i) for i in list(args.unique_pair_scaling.split(","))]
   
    #Consistency checks for input
    check_topologies(pdbs_CG) # Check if input conformations have same topology
    verify_scaling_inputs(quaternary_energy_scaling, pdbs_CG[0].n_chains,input_conformations,unique_pair_scaling)

    #Lists for collection of data
    all_secondary_pairs = []
    all_secondary_pairs_dists_energies = []
    all_tertiary_pairs = []
    all_tertiary_pairs_dists_energies = []

    # Create filters for the first conformation
    filters = create_filters(pdbs_CG[0])
    a_ndx_to_bead_name, a_ndx_to_res_name, a_ndx_to_res_ndx, a_ndx_to_chain_ndx = filters

    #Interate through all input conformations and store the pair information seperately
    for p,pdb_CG in enumerate(pdbs_CG):
        print("Processing conformation {}".format(input_conformations[p]))
        print("Computing distances between all pairs...")
        #Find all possible bead pairs
        all_pairs = np.array([[i,j] for i in np.arange(pdb_CG.top.n_atoms) for j in np.arange(i+1,pdb_CG.top.n_atoms)])
        all_dists = md.compute_distances(pdb_CG,all_pairs)[0]

        secondary_pairs,secondary_dists,tertiary_pairs,tertiary_dists = knowledge_based_checks(all_pairs,all_dists,filters)
        
        cut_secondary_pairs, cut_secondary_dists = apply_cutoff(secondary_pairs, secondary_dists, secondary_cutoff)
        cut_tertiary_pairs, cut_tertiary_dists = apply_cutoff(tertiary_pairs, tertiary_dists, tertiary_cutoff)
    
        #Check if HB pairs
        checked_secondary_pairs,checked_secondary_pairs_dists_energies = check_HB_pairs(cut_secondary_pairs,cut_secondary_dists,hbond_dict,energy_dict,filters)
        checked_tertiary_pairs,checked_tertiary_pairs_dists_energies = check_HB_pairs(cut_tertiary_pairs,cut_tertiary_dists,hbond_dict,energy_dict,filters)

        # Match networks
        matched_secondary_pairs, matched_secondary_pairs_dists_energies = match_secondary_network(checked_secondary_pairs_dists_energies, checked_secondary_pairs)
        matched_tertiary_pairs, matched_tertiary_pairs_dists_energies = match_tertiary_network(checked_tertiary_pairs_dists_energies, checked_tertiary_pairs, a_ndx_to_res_name, a_ndx_to_bead_name, special_residues)
    
        #collect data from each conformation
        all_secondary_pairs.append(matched_secondary_pairs)  
        all_secondary_pairs_dists_energies.append(matched_secondary_pairs_dists_energies)
        all_tertiary_pairs.append(matched_tertiary_pairs)  
        all_tertiary_pairs_dists_energies.append(matched_tertiary_pairs_dists_energies)
        
        if not args.silent:
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

    #Write output files
    generate_output(args, itp_CG, secondary_exclusions, secondary_LJ_potentials, tertiary_LJ_potentials, secondary_harm_potentials, tertiary_harm_potentials, info_secondary_pairs, info_tertiary_pairs, secondary_quaternary_pair_info, tertiary_quaternary_pair_info, quaternary_energy_scaling)


    if not args.silent:
        print('Remember to use "gmx mdrun --noddcheck -rdd 2.0" to avoid domain decomposition warnings if OLIVES bonds gets too long.')
        print('"-noddcheck" turns off a GROMACS domain decomposition error for pairs that become too long relative to the length of a domain (set by -rdd), abruptly ending your run.')
        print('This could happen if a protein complex dissociates or a protein unfolds. The 2 nm cutoff for domains in -rdd is where a LJ potential with energy minimum distance at 0.55 nm (the OLIVES cutoff) goes to 0, and therefore not important if missed.')
        print("If you experience DD errors that cannot be solved by --noddcheck, try using 1 MPI tread and all openMP threads instead of domain decomposition https://manual.gromacs.org/documentation/current/user-guide/mdrun-performance.html.")
        print("OLIVES has been tested using the -scfix flag of martinize2 - if you are using multistate OLIVES, remember to combine the -scfix flag with the OLIVES scfix script")
        print("OLIVES does not use the virtual site approach implemented in e.g. GoMARTINI and thus does not use the martinize2 flags -govs-include.")
        print("OLIVES does not require -dssp secondary structure restraints, although one can opt to restrain the secondary structure at the cost of overstabilized secondary structure.")
    
    print(" o o o o o o o o o ")
    print("Done!")

if __name__ == '__main__':
    main()

