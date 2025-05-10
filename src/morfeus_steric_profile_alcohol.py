"""
File: morfeus_steric_profile_alcohol.py
Author: Hideya Tanaka
Reviewer: Tomoyuki Miyao
Description:
    This script performs Morfeus-based calculations of buried volume at 
    multiple radii. It then outputs PNG figures showing the steric profile. 
    The user can specify a list of radii as input arguments. If the user 
    sets an optional 'radius_for_stericmap', the script will also 
    generate steric map figures for that radius.
"""

import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from joblib import cpu_count, delayed, Parallel
import matplotlib.pyplot as plt
from morfeus import read_xyz, BuriedVolume

def plot_steric_profile(radius_list, buried_volume_list, png_file_path_output, figure_title):
    plt.rcParams.update({'font.size': 15})
    fig, ax_left = plt.subplots()
    ax_left.scatter(radius_list, buried_volume_list)
    ax_left.set_xlabel('Sphere radius (Å)')
    ax_left.set_ylabel('Buried volume (Å³)')
    fig.suptitle(figure_title)
    fig.tight_layout()
    fig.savefig(png_file_path_output, dpi=300)
    plt.close(fig)

def extract_value_from_nested_dict(nested_dict, outer_key, inner_key):
    inner_dict = nested_dict.get(outer_key)
    if inner_dict is None:
        return None
    return inner_dict.get(inner_key)

def calc_morfeus_qo(
        parallel_id, batch_df, fd, xyz_dir_path_input, 
        keyword, radius_list, radius_for_stericmap):
    stericmap_dir_path_output = f'{fd}/morfeus_fig_exclude_H_{keyword}'
    os.makedirs(stericmap_dir_path_output, exist_ok=True)

    ntotal = len(batch_df)
    for worker_id, row in enumerate(batch_df.itertuples(index=True)):
        if parallel_id == 0:
            with open(f'{fd}/log_morfeus_worker_{keyword}.txt', 'w') as f:
                print(f'Processing {worker_id+1}/{ntotal}', file=f)
            
        number = row.Index
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
        bv_exclude_Hs_E_buried_list = []
        bv_exclude_Hs_W_buried_list = []
        for radius in radius_list:
            # Quadrants and octants buried volume
            bv_exclude_Hs = BuriedVolume(elements, coordinates, idx_center, include_hs=False, radius=radius, z_axis_atoms=[idx_z_axis], xz_plane_atoms=idx_xz_plane_list)
            bv_exclude_Hs.octant_analysis()
            bv_exclude_Hs_quadrant = bv_exclude_Hs.quadrants
            
            bv_exclude_Hs_NE_buried = extract_value_from_nested_dict(bv_exclude_Hs_quadrant, 'buried_volume', 1)
            bv_exclude_Hs_NW_buried = extract_value_from_nested_dict(bv_exclude_Hs_quadrant, 'buried_volume', 2)
            bv_exclude_Hs_SW_buried = extract_value_from_nested_dict(bv_exclude_Hs_quadrant, 'buried_volume', 3)
            bv_exclude_Hs_SE_buried = extract_value_from_nested_dict(bv_exclude_Hs_quadrant, 'buried_volume', 4)
            
            bv_exclude_Hs_E_buried_list.append(bv_exclude_Hs_NE_buried + bv_exclude_Hs_SE_buried)
            bv_exclude_Hs_W_buried_list.append(bv_exclude_Hs_NW_buried + bv_exclude_Hs_SW_buried)
                        
            # Drawing the steric map of %Vbur
            if radius_for_stericmap == radius:
                steric_map_file_path_output = f'{stericmap_dir_path_output}/stericmap_{number}_{keyword}_{radius_for_stericmap}.png'
                bv_exclude_Hs.plot_steric_map(filename=steric_map_file_path_output)
            
        # Plot the steric profile
        png_file_path_output = f'{stericmap_dir_path_output}/steric_profile_east_{number}_{keyword}.png'
        figure_title = f'steric_profile_east_{number}'
        plot_steric_profile(radius_list, bv_exclude_Hs_E_buried_list, png_file_path_output, figure_title)
        png_file_path_output = f'{stericmap_dir_path_output}/steric_profile_west_{number}_{keyword}.png'
        figure_title = f'steric_profile_west_{number}'
        plot_steric_profile(radius_list, bv_exclude_Hs_W_buried_list, png_file_path_output, figure_title)
            
def load_mols_from_sdf(fname, remove_Hs=False):
    suppl = Chem.SDMolSupplier(fname, removeHs=remove_Hs)
    mol_list = [mol for mol in suppl if mol is not None]
    print(f'All molecules successfully converted from the SDF file {len(mol_list)}')
    return mol_list

def process_rows_for_morfeus_profile(
        fd, dataset_file_path_input, xyz_dir_path_input, sdf_file_path_input,  
        keyword, radius_list, radius_for_stericmap, njobs=-1):
    print('Morfeus calculation')
    print('========== Settings ==========')
    print(f'keyword: {keyword}')
    print(f'radius_list: {radius_list}')
    print(f'radius_for_stericmap: {radius_for_stericmap}')
    print(f'dataset_file_path_input: {dataset_file_path_input}')
    print(f'xyz_dir_path_input: {xyz_dir_path_input}')
    print(f'sdf_file_path_input: {sdf_file_path_input}')
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

    Parallel(n_jobs=njobs, backend='loky')([delayed(calc_morfeus_qo)
                                            (parallel_id, batch_df, fd, xyz_dir_path_input, 
                                            keyword, radius_list, radius_for_stericmap) 
                                            for parallel_id, batch_df in enumerate(batch_df_list)])

if __name__ == '__main__':
    fd = os.path.dirname(os.path.abspath(__file__))
    
    # ========== Settings (CHANGE HERE) ==========    
    # A name used to identify output directories and files (Modification is not mandatory)
    keyword = f'center_C_zaxis_O_xzplane_H(C)_sp'

    # List of radii used for calculations in Morfeus
    radius_list = [round(r, 1) for r in np.arange(1.5, 15.1, 0.1)]
    
    # A specific radius used for drawing the steric map of %Vbur
    # If you do not need the steric map, set it to 0
    radius_for_stericmap = 10
    
    # Enter the file and directory paths
    dataset_file_path_input = f'{fd}/morfeus_center_H(C)_attached_C.csv'
    xyz_dir_path_input = f'{fd}/gaussian_optimized_xyz'
    sdf_file_path_input = f'{fd}/gaussian_optimized_mols.sdf'
    # =============================================

    sys.stdout = open(f'{fd}/log_morfeus_{keyword}.txt', 'w')
    process_rows_for_morfeus_profile(fd, dataset_file_path_input, xyz_dir_path_input, sdf_file_path_input,  
                                keyword, radius_list, radius_for_stericmap, njobs=-1)
    print('Finish')
    sys.stdout.close()
