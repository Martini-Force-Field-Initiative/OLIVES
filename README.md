# OLIVES: A Go-like Model for Stabilizing Protein Structure via Hydrogen Bonding Beads in the Martini 3 Coarse-Grained Force Field

What is a Martini without OLIVES?

OLIVES is the name of an algorithm that identifices hydrogen bond networks in coarse-grained 
protein structures which are used to implement a Go-like for Martini 3 proteins.

OLIVES enables simulations of Martini 3 proteins without the use if DSSP defined secondary structure restraints,
allowing for more realistic flexibility and at the same time speeds up simulations due to fewer bias potentials. 
The Go-like model has been validated for a range of protein complexes as described in Pedersen et al. (2024). 

## Citation

Pedersen et al. (2024). TBD

## Installation

OLIVES require python 3.7 or greater, together with numpy, mdtraj, and networkx, which can be installed with pip:

    pip install numpy mdtraj networkx

The exact versions should not matter too much, but install networkx 2.3 or greater.
To install exact package versions used during development, run:
	
    pip install numpy==1.21.5 mdtraj==1.9.7 networkx==2.3

## Basic usage

The OLIVES script can be called from the command line. The options of the program 
can be viewed by running:

    python3 OLIVES_v1.0_M3.0.0.py -h

The following command line prompt will convert an atomistic monomeric protein to a coarse-grained representation using martinize2,
generating a topology (with default name molecule_0.itp);

    martinize2 -f protein.pdb -x protein_CG.pdb -o protein_CG.top -scfix -cys auto

and then apply the OLIVES model via:

    python3 OLIVES_v1.0_M3.0.0.py -c "protein_CG.pdb" -i "molecule_0.itp"

This will automatically insert the OLIVES model into the molecule_0.itp topology. There is also an option to write out the OLIVES model in a separate .itp file.
Note that we have left out the -dssp/-ss flags of martinize2 to avoid generating secondary structure restraints. 
Secondary structure restrains could be included using -dssp/-ss flags, if a static secondary structure is desired.
OLIVES was tested using the -scfix flag, although the side chains conformations are also influenced by the OLIVES LJ potentials. 
Additional information files about the network can be written, see the the help command (-h). 
The generated OLIVES pairs could be used to drive biased simulations due to the similarity to native contacts. 

OLIVES also comes with a basic multistate functionality. A two-state model can be created by providing two conformations of the same protein (must have matching topologies):

    python3 OLIVES_v1.0_M3.0.0.py -c "protein_CG_conformation_1.pdb,protein_CG_conformation_2.pdb" -i molecule_0.itp --unique_pair_scaling "0.5,0.75"

The enthalpy of contacts unique to each conformations will be scaled by --unique_pair_scaling. In this example the unique contacts for conformation 1 are downscaled by 0.5 and conformation 2 by 0.75.
This can be used to tune the relative free energies between conformations. Try --unique_pair_scaling "0.5,0.5" as an initial guees, if building a model with unknown relative free energies. 
Shared contacts between conformations are not scaled (unless specified by the --ss_h_scaling and --tt_h_scaling flags), but instead have their minimum distance averaged.

More examples on how to set up protein complexes with and without quaternary networks can be found in the tutorials folder in the source repository.

## License

OLIVES and the contents of this repository are distributed under the Apache 2.0 license.

    Copyright 2023 Aarhus University

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

                http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.

The full text of the license is available in the source repository.

## Contributions

The development of OLIVES is done on [github]. If you encounter any problems please file an [issue].
Contributions are welcome as [pull requests]. Note however that the
decision of whether or not contributions can give authorship on resulting
academic work is left to our sole discretion.

[github]: https://github.com/Martini-Force-Field-Initiative/OLIVES
[issue]: https://github.com/Martini-Force-Field-Initiative/OLIVES/issues
[pull requests]: https://github.com/Martini-Force-Field-Initiative/OLIVES/pulls



