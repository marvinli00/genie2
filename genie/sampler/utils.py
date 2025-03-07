import os
import numpy as np
from Bio.PDB import PDBParser

def process_pdb(pdb_file_path, atom_filter=None, group_by="ter", atom_properties=None):
    """
    Generic function to process PDB files with customizable behavior.
    
    Arguments:
        pdb_file_path (str): Path to the PDB file to process
        atom_filter (function, optional): Function that takes a line and returns True if the 
                                         atom should be included, False otherwise
        group_by (str, optional): How to group atoms - options:
                                 "ter" - group by TER separations
                                 "chain" - group by chain ID
                                 "residue" - group by residue
                                 "model" - group by MODEL records
                                 "none" - no grouping, return flat list
        atom_properties (list, optional): List of properties to extract for each atom
                                         Default: ['x', 'y', 'z']
    
    Returns:
        dict or list: Processed data based on specified grouping and properties
    """
    if atom_properties is None:
        atom_properties = ['x', 'y', 'z']
    
    # Default filter: include all ATOM records
    if atom_filter is None:
        atom_filter = lambda line: line.startswith("ATOM")
    
    # Property extraction functions
    property_extractors = {
        'x': lambda line: float(line[30:38].strip()),
        'y': lambda line: float(line[38:46].strip()),
        'z': lambda line: float(line[46:54].strip()),
        'atom_name': lambda line: line[12:16].strip(),
        'residue_name': lambda line: line[17:20].strip(),
        'chain_id': lambda line: line[21:22].strip(),
        'residue_number': lambda line: int(line[22:26].strip()),
        'element': lambda line: line[76:78].strip(),
        'b_factor': lambda line: float(line[60:66].strip()),
        'occupancy': lambda line: float(line[54:60].strip()),
        'full_line': lambda line: line
    }
    
    # Initialize result structures
    result = {}
    current_group_key = None
    current_model = "1"
    
    with open(pdb_file_path, 'r') as pdb_file:
        for line in pdb_file:
            # Handle MODEL records for NMR structures
            if line.startswith("MODEL"):
                current_model = line[10:14].strip()
                if group_by == "model":
                    current_group_key = current_model
                    if current_group_key not in result:
                        result[current_group_key] = []
            
            # Process TER records
            elif line.startswith("TER") and group_by == "ter":
                current_group_key = None
            
            # Process ENDMDL records
            elif line.startswith("ENDMDL") and group_by == "model":
                current_group_key = None
            
            # Process atoms
            elif atom_filter(line):
                # Extract atom properties
                atom_data = {}
                for prop in atom_properties:
                    if prop in property_extractors:
                        atom_data[prop] = property_extractors[prop](line)
                    else:
                        atom_data[prop] = None  # Property extractor not defined
                
                # Determine grouping
                if group_by == "ter":
                    if current_group_key is None:
                        current_group_key = len(result)
                        result[current_group_key] = []
                
                elif group_by == "chain":
                    current_group_key = property_extractors['chain_id'](line)
                    if current_group_key not in result:
                        result[current_group_key] = []
                
                elif group_by == "residue":
                    residue_id = (
                        property_extractors['chain_id'](line),
                        property_extractors['residue_number'](line),
                        property_extractors['residue_name'](line)
                    )
                    current_group_key = residue_id
                    if current_group_key not in result:
                        result[current_group_key] = []
                
                elif group_by == "model":
                    if current_group_key is None:
                        current_group_key = current_model
                        if current_group_key not in result:
                            result[current_group_key] = []
                
                elif group_by == "none":
                    if "atoms" not in result:
                        result["atoms"] = []
                    current_group_key = "atoms"
                
                # Add atom data to the appropriate group
                if len(atom_properties) == 1:
                    result[current_group_key].append(atom_data[atom_properties[0]])
                else:
                    result[current_group_key].append(atom_data)
    
    # Convert to list if requested
    if group_by == "none":
        return result["atoms"]
    
    # Convert ordered result for some grouping types
    if group_by == "ter":
        return [result[i] for i in sorted(result.keys())]
    
    return result



def load_motif_target(index, folder_path = "/hai/scratch/mli89/protein_design/MotifBench/motif_pdbs"):    
    #write a function to read the motif segments from the pdb file
    #read files in the folder 
    folder_path = "/hai/scratch/mli89/protein_design/MotifBench/motif_pdbs"
    #motif_target = []
    files = os.listdir(folder_path)
    #sort the files by the number
    files = sorted(files, key=lambda x: int(x.split('_')[0]))
    
    path = os.path.join(folder_path, files[index])
    ca_filter = lambda line: line.startswith("ATOM") and line[12:16].strip() == "CA"
    coords = process_pdb(path, atom_filter=ca_filter, group_by="ter")
    coords_flatten = process_pdb(path, atom_filter=ca_filter, group_by="none")
    
    #find center of mass from coords_filtered
    mean_of_coordinates = lambda x, name_of_coordinate: np.mean([i[name_of_coordinate] for i in x])
    x_mean = mean_of_coordinates(coords_flatten, 'x')
    y_mean = mean_of_coordinates(coords_flatten, 'y')
    z_mean = mean_of_coordinates(coords_flatten, 'z')
    
    coords_to_numpy = []
    
    #convert 
    if len(coords_flatten) == len(coords):
        coords_segment = []
        for i in range(len(coords)):
            coords[i]['x'] = coords[i]['x'] - x_mean
            coords[i]['y'] = coords[i]['y'] - y_mean
            coords[i]['z'] = coords[i]['z'] - z_mean
            coords_segment.append(np.stack((coords[i]['x'], coords[i]['y'], coords[i]['z'])))
        coords_to_numpy.append(coords_segment.reshape(-1,3))
        #motif_target.append(coords)
    else:
        #center each of segement in coords
        for i in range(len(coords)):
            coords_segment = []
            for j in range(len(coords[i])):
                coords[i][j]['x'] = coords[i][j]['x'] - x_mean
                coords[i][j]['y'] = coords[i][j]['y'] - y_mean
                coords[i][j]['z'] = coords[i][j]['z'] - z_mean
                coords_segment.append(np.stack((coords[i][j]['x'], coords[i][j]['y'], coords[i][j]['z'])))
            coords_to_numpy.append(np.stack(coords_segment, axis = 0))
        #motif_target.append(coords)

    return coords_to_numpy