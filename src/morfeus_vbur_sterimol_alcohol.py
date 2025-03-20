"""
File: morfeus_vbur_sterimol_alcohol.py
Author: Hideya Tanaka
Reviewer: Tomoyuki Miyao
Description:
    This script performs Morfeus-based steric parameter calculations
    (e.g., %Vbur and Sterimol) on primary or secondary alcohol molecules.
    By specifying center_atom and attached_atom as C, O, H(O), or H(C),
    the script tailors the calculations to focus on the desired atoms
    within each molecule. The input .xyz files are generated from 
    Gaussian-optimized structures, and matching .sdf files with the 
    same atomic coordinates are used to retrieve consistent atom 
    indices for Morfeus.
"""

import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from joblib import cpu_count, delayed, Parallel
from morfeus import read_xyz, BuriedVolume, Sterimol

def calc_morfeus(
        parallel_id, batch_df, fd, xyz_dir_path_input, center_atom, attached_atom, 
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
        if center_atom == 'H(C)':
            smarts_query = '[H]-[C;!H0;X4]-[O;H1;X2]'
        else:
            smarts_query = '[C;!H0;X4]-[O;H1;X2]-[H]'
        mol_query = Chem.MolFromSmarts(smarts_query)
        gsm = mol.GetSubstructMatches(mol_query)
        idx_HC_morfeus_list = []

        # Identify atom indices to be used in Morfeus calculations
        if center_atom != 'H(C)':
            if len(gsm) != 1:
                raise ValueError('The gsm elements must be 1.')
        else:
            if len(gsm) > 2:
                raise ValueError('The gsm elements must not exceed 2.')
        for each_gsm in gsm:
            for aidx in each_gsm:
                atomicnum = mol.GetAtomWithIdx(aidx).GetAtomicNum()
                if  atomicnum == 6:
                    idx_C_morfeus_list = [aidx + 1] # RDKit indices are '0-indexed', while Morfeus indices are '1-indexed'
                elif atomicnum == 8:
                    idx_O_morfeus_list = [aidx + 1]
                elif atomicnum == 1 and center_atom != 'H(C)':
                    idx_HO_morfeus_list = [aidx + 1]
                elif atomicnum == 1 and center_atom == 'H(C)':
                    idx_HC_morfeus = aidx + 1
                    idx_HC_morfeus_list.append(idx_HC_morfeus)
                else:
                    raise ValueError('Either C, O, or H must be present.')
                    
        if  center_atom == 'C':
            idx_center_list = idx_C_morfeus_list
        elif center_atom == 'O':
            idx_center_list = idx_O_morfeus_list
        elif center_atom == 'H(O)':
            idx_center_list = idx_HO_morfeus_list
        elif center_atom == 'H(C)':
            idx_center_list = idx_HC_morfeus_list
        else:
            raise ValueError('Set either C, O, H(O), or H(C) as the center_atom.')
            
        if  attached_atom == 'C':
            idx_attached = idx_C_morfeus_list[0]
        elif attached_atom == 'O':
            idx_attached = idx_O_morfeus_list[0]
        elif attached_atom == 'H(O)':
            idx_attached = idx_HO_morfeus_list[0]
        elif attached_atom == 'H(C)':
            raise ValueError('Do not set H(C) as attached_atom because some molecules have two H(C) atoms.')
        else:
            raise ValueError('Set either C, O, or H(O) as the attached_atom.')
            
        # Morfeus calculations
        sparam_dict = dict()
        sparam_dict['smiles_morfeus'] = smiles
        for radius in radius_list:
            bv_include_Hs_list = []
            bv_exclude_Hs_list = []
            bs_B1_list = []
            bs_B5_list = []
            bs_L_list = []
            s_B1_list = []
            s_B5_list = []
            s_L_list = []
            for idx_center in idx_center_list:
                # Buried volume
                bv_include_Hs = BuriedVolume(elements, coordinates, idx_center, include_hs=True, radius=radius, z_axis_atoms=[idx_attached])
                bv_include_Hs_list.append(bv_include_Hs.fraction_buried_volume * 100)
                bv_exclude_Hs = BuriedVolume(elements, coordinates, idx_center, include_hs=False, radius=radius, z_axis_atoms=[idx_attached])
                bv_exclude_Hs_list.append(bv_exclude_Hs.fraction_buried_volume * 100)
                
                # Sterimol
                sterimol = Sterimol(elements, coordinates, idx_center, idx_attached)
                s_B1_list.append(sterimol.B_1_value)
                s_B5_list.append(sterimol.B_5_value)
                s_L_list.append(sterimol.L_value)
                
                # Buried sterimol
                sterimol.bury(sphere_radius=radius, method="delete")
                bs_B1_list.append(sterimol.B_1_value)
                bs_B5_list.append(sterimol.B_5_value)
                bs_L_list.append(sterimol.L_value)
            
            # Drawing the steric map of %Vbur
            if radius_for_stericmap == radius:
                steric_map_file_path_output = f'{stericmap_dir_path_output}/{number}_stericmap_{keyword}_{radius_for_stericmap}.png'
                bv_exclude_Hs.plot_steric_map(filename=steric_map_file_path_output)
                
            if center_atom != 'H(C)':
                sparam_dict[f'%Vbur_include_H_center_{center_atom}_{radius}'] = round(bv_include_Hs_list[0], 3)
                sparam_dict[f'%Vbur_exclude_H_center_{center_atom}_{radius}'] = round(bv_exclude_Hs_list[0], 3)
                sparam_dict[f'buried_sterimol_B1_{keyword}_{radius}'] = round(bs_B1_list[0], 3)
                sparam_dict[f'buried_sterimol_B5_{keyword}_{radius}'] = round(bs_B5_list[0], 3)
                sparam_dict[f'buried_sterimol_L_{keyword}_{radius}'] = round(bs_L_list[0], 3)
            else:
                sparam_dict[f'%Vbur_include_H_center_{center_atom}_{radius}_mean'] = round(np.mean(bv_include_Hs_list), 3)
                sparam_dict[f'%Vbur_include_H_center_{center_atom}_{radius}_min'] = round(min(bv_include_Hs_list), 3)
                sparam_dict[f'%Vbur_include_H_center_{center_atom}_{radius}_max'] = round(max(bv_include_Hs_list), 3)
                
                sparam_dict[f'%Vbur_exclude_H_center_{center_atom}_{radius}_mean'] = round(np.mean(bv_exclude_Hs_list), 3)
                sparam_dict[f'%Vbur_exclude_H_center_{center_atom}_{radius}_min'] = round(min(bv_exclude_Hs_list), 3)
                sparam_dict[f'%Vbur_exclude_H_center_{center_atom}_{radius}_max'] = round(max(bv_exclude_Hs_list), 3)
                
                sparam_dict[f'buried_sterimol_B1_{keyword}_{radius}_mean'] = round(np.mean(bs_B1_list), 3)
                sparam_dict[f'buried_sterimol_B1_{keyword}_{radius}_min'] = round(min(bs_B1_list), 3)
                sparam_dict[f'buried_sterimol_B1_{keyword}_{radius}_max'] = round(max(bs_B1_list), 3)
                
                sparam_dict[f'buried_sterimol_B5_{keyword}_{radius}_mean'] = round(np.mean(bs_B5_list), 3)
                sparam_dict[f'buried_sterimol_B5_{keyword}_{radius}_min'] = round(min(bs_B5_list), 3)
                sparam_dict[f'buried_sterimol_B5_{keyword}_{radius}_max'] = round(max(bs_B5_list), 3)
                
                sparam_dict[f'buried_sterimol_L_{keyword}_{radius}_mean'] = round(np.mean(bs_L_list), 3)
                sparam_dict[f'buried_sterimol_L_{keyword}_{radius}_min'] = round(min(bs_L_list), 3)
                sparam_dict[f'buried_sterimol_L_{keyword}_{radius}_max'] = round(max(bs_L_list), 3)
                
        if center_atom != 'H(C)':
            sparam_dict[f'sterimol_B1_{keyword}'] = round(s_B1_list[0], 3)
            sparam_dict[f'sterimol_B5_{keyword}'] = round(s_B5_list[0], 3)
            sparam_dict[f'sterimol_L_{keyword}'] = round(s_L_list[0], 3)
        else:
            sparam_dict[f'sterimol_B1_{keyword}_mean'] = round(np.mean(s_B1_list), 3)
            sparam_dict[f'sterimol_B1_{keyword}_min'] = round(min(s_B1_list), 3)
            sparam_dict[f'sterimol_B1_{keyword}_max'] = round(max(s_B1_list), 3)
            
            sparam_dict[f'sterimol_B5_{keyword}_mean'] = round(np.mean(s_B5_list), 3)
            sparam_dict[f'sterimol_B5_{keyword}_min'] = round(min(s_B5_list), 3)
            sparam_dict[f'sterimol_B5_{keyword}_max'] = round(max(s_B5_list), 3)
            
            sparam_dict[f'sterimol_L_{keyword}_mean'] = round(np.mean(s_L_list), 3)
            sparam_dict[f'sterimol_L_{keyword}_min'] = round(min(s_L_list), 3)
            sparam_dict[f'sterimol_L_{keyword}_max'] = round(max(s_L_list), 3)
        
        steric_param_dict[number] = sparam_dict
        
    return pd.DataFrame.from_dict(steric_param_dict, orient='index')

def load_mols_from_sdf(fname, remove_Hs=False):
    suppl = Chem.SDMolSupplier(fname, removeHs=remove_Hs)
    mol_list = [mol for mol in suppl if mol is not None]
    print(f'All molecules successfully converted from the SDF file {len(mol_list)}')
    return mol_list

def process_rows_for_morfeus(
        fd, dataset_file_path_input, xyz_dir_path_input, sdf_file_path_input, dataset_file_path_output, 
        center_atom, attached_atom, keyword, radius_list, radius_for_stericmap, njobs=-1):
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

    output_df_list = Parallel(n_jobs=njobs, backend='loky')([delayed(calc_morfeus)
                                                             (parallel_id, batch_df, fd, xyz_dir_path_input, center_atom, 
                                                              attached_atom, keyword, radius_list, radius_for_stericmap) 
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
    # Specify the central atom here for %Vbur and sterimol calculations in Morfeus
    center_atom = 'C' # Input either C, O, H(O), or H(C)
    
    # Specify the atom here to define the vector for the L length direction in sterimol calculations
    # This atom is also used to define the z-axis for generating the steric map of %Vbur
    attached_atom = 'O' # Input either C, O, or H(O)
    
    # A name used to identify output directories and files (Modification is not mandatory)
    keyword = f'center_{center_atom}_attached_{attached_atom}'

    # List of radii used for calculations in Morfeus
    radius_list = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
    
    # A specific radius used for drawing the steric map of %Vbur
    # If you do not need the steric map, set it to 0
    radius_for_stericmap = 2.5
    
    # Enter the file and directory paths
    dataset_file_path_input = f'{fd}/gaussian_optimized_alcohols.csv'
    xyz_dir_path_input = f'{fd}/gaussian_optimized_xyz'
    sdf_file_path_input = f'{fd}/gaussian_optimized_mols.sdf'
    dataset_file_path_output = f'{fd}/morfeus_{keyword}.csv'
    # =============================================

    sys.stdout = open(f'{fd}/log_morfeus_{keyword}.txt', 'w')
    process_rows_for_morfeus(fd, dataset_file_path_input, xyz_dir_path_input, sdf_file_path_input, dataset_file_path_output, 
                             center_atom, attached_atom, keyword, radius_list, radius_for_stericmap, njobs=-1)
    print('Finish')
    sys.stdout.close()
