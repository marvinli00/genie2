from biotite.structure import annotate_sse
import biotite.structure.io.pdb as pdb
import numpy as np

import sys
sys.path.append("../genie2/")



import re
import os

def fix_pdb_columns(pdb_path):
    """
    Reads a PDB file, fixes missing occupancy and B-factor columns,
    and writes a corrected file with '_fixed' appended to the filename.

    Parameters:
    pdb_path (str): Path to the input PDB file.
    """
    fixed_pdb_path = pdb_path.replace(".pdb", "_fixed.pdb")

    with open(pdb_path, "r") as infile, open(fixed_pdb_path, "w") as outfile:
        for line in infile:
            if line.startswith(("ATOM", "HETATM")):
                # Ensure correct formatting using PDB column specifications
                # Occupancy (54-59), B-factor (60-65)
                fixed_line = line[:54].ljust(54) + " 1.00  0.00" + line[66:]
                outfile.write(fixed_line + "\n")
            else:
                outfile.write(line)  # Write other lines unchanged
    return fixed_pdb_path

def get_coords_from_pdb(pdb_path):
    # Convert to AtomArray (or AtomArrayStack for multiple models)
    fixed_pdb_path = fix_pdb_columns(pdb_path)
    pdb_file = pdb.PDBFile.read(fixed_pdb_path)
    atom_array = pdb.get_structure(pdb_file)
    array = atom_array.get_array(0).coord
    return array

def sec_struct_frac(pdb_path):
    """sec_struct_frac returns the fraction of residues of the pdb in [helix,
    strand, coil]
    """

    fixed_pdb_path = fix_pdb_columns(pdb_path)
    pdb_file = pdb.PDBFile.read(fixed_pdb_path)

    # Convert to AtomArray (or AtomArrayStack for multiple models)
    atom_array = pdb.get_structure(pdb_file)

    # Remove fixed pdb file
    os.remove(fixed_pdb_path)

    sse = annotate_sse(atom_array[0])
    return np.array([sum([ss_val == ss_type for ss_val in sse] ) for ss_type
        in ['a', 'b', 'c']])/len(sse)


from genie.utils.feat_utils import save_np_features_to_pdb
def save_pdb(x, pdb_path):
    """save_pdb write a Ca only pdb file using Genie2 utility function.
    
    Args:
        x: [N, 3] numpy array, units of Angstroms
    """
    assert x.ndim == 2 and x.shape[1] == 3
    N = x.shape[0]
    np_features = {
        "atom_positions":x,
        "aatype": np.array([np.eye(N)[0] for _ in range(N)]),
        "residue_index": np.arange(N),
        "chain_index": np.zeros(N, dtype=int),
        "fixed_group": np.zeros(N, dtype=int)
        }
    save_np_features_to_pdb(np_features, pdb_path)
    

import random
import string
def h(x):
    """h is an example statistic.  
        h(x) = 1 if the structure is > 50% alpha-helix and is 0 otherwise.

    Args:
        x: np array of shape [N, 3], units of Angstroms.

    Returns:
        scalar 0 or 1
    """
    random_pdb_path = ''.join(random.choices(string.ascii_letters + string.digits,
        k=10)) + '.pdb'
    save_pdb(x, random_pdb_path)
    ss_frac = sec_struct_frac(random_pdb_path)[0] > 0.5
    os.remove(random_pdb_path)
    return ss_frac

if __name__ == "__main__":
    pdb_path = "/home/users/btrippe/projects/diffusion_calibration/genie2/results/base/outputs/pdbs/50_0.pdb"
    ss_frac = sec_struct_frac(pdb_path)
    print("\t".join([f"{100*v}% {ss_type}" for (ss_type, v) in zip(["helix",
        "strand", "coil"], list(ss_frac))]))

    ## test going through coordinates
    coords = get_coords_from_pdb(pdb_path)
    h_val = h(coords)
    print("h_val:", h_val)