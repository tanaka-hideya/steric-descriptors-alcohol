"""
File: morfeus_vbur_quadrant_octant_alcohol.py
Author: Hideya Tanaka
Reviewer: Tomoyuki Miyao
Description:
    This script performs Morfeus-based %Vbur_quadrant and %Vbur_octant
    calculations on primary or secondary alcohol molecules. The center 
    atom is set to carbon (C), the z-axis is defined by an oxygen (O) 
    atom, and a hydrogen attached to the carbon (H(C)) is used for the 
    xz-plane reference. The input .xyz files are generated from 
    Gaussian-optimized structures, and matching .sdf files with the 
    same atomic coordinates are used to retrieve consistent atom indices for Morfeus.
"""

import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from joblib import cpu_count, delayed, Parallel
from morfeus import read_xyz, BuriedVolume

def extract_value_from_nested_dict(nested_dict, outer_key, inner_key):
    inner_dict = nested_dict.get(outer_key)
    if inner_dict is None:
        return None
    return inner_dict.get(inner_key)

def calc_morfeus_qo(
        parallel_id, batch_df, fd, xyz_dir_path_input, 
        keyword, radius_list, radius_for_stericmap):
    if radius_for_stericmap != 0:
        stericmap_dir_path_output = f'{fd}/morfeus_stericmap_exclude_H_{keyword}_{radius_for_stericmap}'
        os.makedirs(stericmap_dir_path_output, exist_ok=True)

    ntotal = len(batch_df)
    steric_param_dict = dict()
    for worker_id, row in enumerate(batch_df.itertuples(index=True)):
        if parallel_id == 0:
            with open(f'{fd}/log_morfeus_worker_{keyword}.txt', 'w') as f:
                print(f'Processing {worker_id+1}/{ntotal}', file=f)
            
        number = row.Index
        smiles = row.smiles
        filepath = row.filepath
        mol = row.mol
        
        # The 'filepath' column indicates the names of Gaussian log files
        # XYZ files created from Gaussian opt and freq log files have filenames corresponding to the log files
        filepath = filepath.replace('.log', '.xyz')
        elements, coordinates = read_xyz(f'{xyz_dir_path_input}/{filepath}')

        # Verify whether the atoms and their coordinates are aligned between the XYZ file and the SDF file
        natoms = mol.GetNumAtoms()
        for idx in range(natoms):
            if (elements[idx] != mol.GetAtomWithIdx(idx).GetSymbol()) or (not np.allclose(coordinates[idx], mol.GetConformer().GetPositions()[idx], atol=1e-3)):
                raise ValueError('The atom index order in Morfeus and RDKit is not consistent.')
        
        # A specific query used to specify atoms in Morfeus calculations
        smarts_query = '[H]-[C;!H0;X4]-[O;H1;X2]'
        mol_query = Chem.MolFromSmarts(smarts_query)
        gsm = mol.GetSubstructMatches(mol_query)
        idx_xz_plane_list = []

        # Identify atom indices to be used in Morfeus calculations
        if len(gsm) > 2:
            raise ValueError('The gsm elements must not exceed 2.')
        for each_gsm in gsm:
            for aidx in each_gsm:
                atomicnum = mol.GetAtomWithIdx(aidx).GetAtomicNum()
                if  atomicnum == 6:
                    idx_center = aidx + 1 # RDKit indices are '0-indexed', while Morfeus indices are '1-indexed'
                elif atomicnum == 8:
                    idx_z_axis = aidx + 1
                elif atomicnum == 1:
                    idx_xz_plane = aidx + 1
                    idx_xz_plane_list.append(idx_xz_plane)
                else:
                    raise ValueError('Either C, O, or H must be present.')
            
        # Morfeus calculations
        sparam_dict = dict()
        sparam_dict['smiles_morfeus'] = smiles
        for radius in radius_list:
            # We are intentionally keeping the list-based approach here for potential future expansions.
            # Although a single variable would suffice at present, this structure helps preserve flexibility for additional loop-based logic in subsequent revisions.
            bv_exclude_Hs_quadrant_E_list = []
            bv_exclude_Hs_quadrant_W_list = []
            bv_exclude_Hs_octant_E_plus_list = []
            bv_exclude_Hs_octant_E_minus_list = []
            bv_exclude_Hs_octant_W_plus_list = []
            bv_exclude_Hs_octant_W_minus_list = []
            # Quadrants and octants buried volume
            bv_exclude_Hs = BuriedVolume(elements, coordinates, idx_center, include_hs=False, radius=radius, z_axis_atoms=[idx_z_axis], xz_plane_atoms=idx_xz_plane_list)
            bv_exclude_Hs.octant_analysis()
            bv_exclude_Hs_quadrant = bv_exclude_Hs.quadrants
            bv_exclude_Hs_quadrant_NE = extract_value_from_nested_dict(bv_exclude_Hs_quadrant, 'percent_buried_volume', 1)
            bv_exclude_Hs_quadrant_NW = extract_value_from_nested_dict(bv_exclude_Hs_quadrant, 'percent_buried_volume', 2)
            bv_exclude_Hs_quadrant_SW = extract_value_from_nested_dict(bv_exclude_Hs_quadrant, 'percent_buried_volume', 3)
            bv_exclude_Hs_quadrant_SE = extract_value_from_nested_dict(bv_exclude_Hs_quadrant, 'percent_buried_volume', 4)
            
            bv_exclude_Hs_quadrant_E_list.append(bv_exclude_Hs_quadrant_NE + bv_exclude_Hs_quadrant_SE)
            bv_exclude_Hs_quadrant_W_list.append(bv_exclude_Hs_quadrant_NW + bv_exclude_Hs_quadrant_SW)
            
            bv_exclude_Hs_octants = bv_exclude_Hs.octants
            bv_exclude_Hs_octant_E_plus_list.append(extract_value_from_nested_dict(bv_exclude_Hs_octants, 'percent_buried_volume', 0)
                                                    + extract_value_from_nested_dict(bv_exclude_Hs_octants, 'percent_buried_volume', 3))
            bv_exclude_Hs_octant_E_minus_list.append(extract_value_from_nested_dict(bv_exclude_Hs_octants, 'percent_buried_volume', 4)
                                                        + extract_value_from_nested_dict(bv_exclude_Hs_octants, 'percent_buried_volume', 7))
            bv_exclude_Hs_octant_W_plus_list.append(extract_value_from_nested_dict(bv_exclude_Hs_octants, 'percent_buried_volume', 1)
                                                    + extract_value_from_nested_dict(bv_exclude_Hs_octants, 'percent_buried_volume', 2))
            bv_exclude_Hs_octant_W_minus_list.append(extract_value_from_nested_dict(bv_exclude_Hs_octants, 'percent_buried_volume', 5)
                                                        + extract_value_from_nested_dict(bv_exclude_Hs_octants, 'percent_buried_volume', 6))
            
            # Drawing the steric map of %Vbur
            if radius_for_stericmap == radius:
                steric_map_file_path_output = f'{stericmap_dir_path_output}/{number}_stericmap_{keyword}_{radius_for_stericmap}.png'
                bv_exclude_Hs.plot_steric_map(filename=steric_map_file_path_output)
            
            sparam_dict[f'%Vbur_exclude_H_quadrant_E_{keyword}_{radius}'] = round(bv_exclude_Hs_quadrant_E_list[0], 3)            
            sparam_dict[f'%Vbur_exclude_H_quadrant_W_{keyword}_{radius}'] = round(bv_exclude_Hs_quadrant_W_list[0], 3)
            sparam_dict[f'%Vbur_exclude_H_octant_E_plus_{keyword}_{radius}'] = round(bv_exclude_Hs_octant_E_plus_list[0], 3)            
            sparam_dict[f'%Vbur_exclude_H_octant_E_minus_{keyword}_{radius}'] = round(bv_exclude_Hs_octant_E_minus_list[0], 3)
            sparam_dict[f'%Vbur_exclude_H_octant_W_plus_{keyword}_{radius}'] = round(bv_exclude_Hs_octant_W_plus_list[0], 3)
            sparam_dict[f'%Vbur_exclude_H_octant_W_minus_{keyword}_{radius}'] = round(bv_exclude_Hs_octant_W_minus_list[0], 3)

        steric_param_dict[number] = sparam_dict
        
    return pd.DataFrame.from_dict(steric_param_dict, orient='index')

def load_mols_from_sdf(fname, remove_Hs=False):
    suppl = Chem.SDMolSupplier(fname, removeHs=remove_Hs)
    mol_list = [mol for mol in suppl if mol is not None]
    print(f'All molecules successfully converted from the SDF file {len(mol_list)}')
    return mol_list

def process_rows_for_morfeus_qo(
        fd, dataset_file_path_input, xyz_dir_path_input, sdf_file_path_input, dataset_file_path_output, 
        keyword, radius_list, radius_for_stericmap, njobs=-1):
    print('Morfeus calculation')
    print('========== Settings ==========')
    print(f'keyword: {keyword}')
    print(f'radius_list: {radius_list}')
    print(f'radius_for_stericmap: {radius_for_stericmap}')
    print(f'dataset_file_path_input: {dataset_file_path_input}')
    print(f'xyz_dir_path_input: {xyz_dir_path_input}')
    print(f'sdf_file_path_input: {sdf_file_path_input}')
    print(f'dataset_file_path_output: {dataset_file_path_output}')
    print('------------------------------')
    
    df = pd.read_csv(dataset_file_path_input, index_col=0)
    print(f'All molecules in the dataset {len(df)}')
    # Assumes the use of a CSV file where the "confid" column contains 0 if Gaussian opt or freq calculations failed
    original_df = df[df['confid'] != 0]
    print(f'All molecules successfully processed by Gaussian opt and freq calculations {len(original_df)}')
    df = original_df.copy()
    
    mol_list = load_mols_from_sdf(sdf_file_path_input)
    df['mol'] = mol_list

    # Parallel calculation setting
    njobs = cpu_count() -1 if njobs < 1 else njobs
    batch_df_list = np.array_split(df, njobs)

    output_df_list = Parallel(n_jobs=njobs, backend='loky')([delayed(calc_morfeus_qo)
                                                             (parallel_id, batch_df, fd, xyz_dir_path_input, 
                                                              keyword, radius_list, radius_for_stericmap) 
                                                             for parallel_id, batch_df in enumerate(batch_df_list)])
    output_df = pd.concat(output_df_list)
    combined_df = pd.concat([original_df, output_df], ignore_index=False, axis=1)
    if combined_df['smiles'].equals(combined_df['smiles_morfeus']):
        combined_df = combined_df.drop(columns='smiles_morfeus')
        print('The newly generated DataFrame was successfully concatenated with the original CSV file.')
    else:
        raise ValueError('The order of rows has changed after parallel computation.')
    combined_df.to_csv(dataset_file_path_output)

if __name__ == '__main__':
    fd = os.path.dirname(os.path.abspath(__file__))
    
    # ========== Settings (CHANGE HERE) ==========    
    # A name used to identify output directories and files (Modification is not mandatory)
    keyword = f'center_C_zaxis_O_xzplane_H(C)'

    # List of radii used for calculations in Morfeus
    radius_list = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
    
    # A specific radius used for drawing the steric map of %Vbur
    # If you do not need the steric map, set it to 0
    radius_for_stericmap = 2.5
    
    # Enter the file and directory paths
    dataset_file_path_input = f'{fd}/morfeus_center_H(C)_attached_C.csv'
    xyz_dir_path_input = f'{fd}/gaussian_optimized_xyz'
    sdf_file_path_input = f'{fd}/gaussian_optimized_mols.sdf'
    dataset_file_path_output = f'{fd}/morfeus_quadrant_octant_{keyword}.csv'
    # =============================================

    sys.stdout = open(f'{fd}/log_morfeus_{keyword}.txt', 'w')
    process_rows_for_morfeus_qo(fd, dataset_file_path_input, xyz_dir_path_input, sdf_file_path_input, dataset_file_path_output, 
                                keyword, radius_list, radius_for_stericmap, njobs=-1)
    print('Finish')
    sys.stdout.close()
