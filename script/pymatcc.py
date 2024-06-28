import os
import sys
import re
import joblib
import pickle
import argparse
from math import gcd
from functools import reduce

import numpy as np
import pandas as pd
from hdbscan import approximate_predict, membership_vector
from pacmap import load
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure

from utils import cprint, pprint, datadir, srcdir, checkexists, plt
from config import ELEMENT, STYPES
from descriptor import evaluate_feature_vector
from oxidation import apply_charge_decoration
from simplification import structure_simplifications
from analysis import plot_manifold, group_analysis


datadir = os.path.join(datadir, ELEMENT)


def to_pretty_formula(composition):
    sub = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
    return composition.translate(sub)


def get_symmetry(structure):
    SGA = SpacegroupAnalyzer(structure)
    return SGA.get_crystal_system()


def extract_elements(x):
    elements = [re.sub(r'[^a-zA-Z]', '', comp) for comp in x.split(' ')]
    assert len(np.unique(elements)) == len(elements)
    return elements


def extract_composition(x):
    numbers = re.findall(r'[-+]?(?:\d*\.*\d+)', x)
    compositions = list(np.array(list(map(float, numbers)))*100000)  # multiplication with a large number to make integer
    compositions = list(map(lambda x: int(x), compositions))
    _gcd = find_gcd(compositions)
    return [int(comp/_gcd) for comp in compositions]


def find_gcd(array):
    x = reduce(gcd, array)
    return x


def extract_structural_information(structure):
    # Retrieve information
    lattice = structure.lattice
    volume = lattice.volume
    density = structure.composition.weight/volume
    material_id = None
    icsd_id = None
    formula = structure.composition.formula.replace(' ', '')
    formula_pretty = to_pretty_formula(formula)
    composition = structure.composition.to_weight_dict
    composition_pretty = structure.composition.formula
    elements = extract_elements(composition_pretty)
    stoichiometry = extract_composition(composition_pretty)
    space_group = structure.get_space_group_info()[0]
    lattice_type = get_symmetry(structure).capitalize()
    structure = apply_charge_decoration(structure)  # decorate with oxidation states
    data = [material_id, icsd_id, formula, formula_pretty, elements, stoichiometry, 
            composition, composition_pretty, space_group, lattice_type, volume, 
            density, lattice, structure]
    
    # Packing up
    ds = pd.Series(data, index=['material_id', 'icsd_id', 'formula', 
                                'formula_pretty', 'elements', 'stoichiometry',
                                'composition', 'composition_pretty', 
                                'space_group', 'lattice_type', 'volume', 
                                'density', 'lattice', 'structure'])
    return ds


def classification(X, clusterer, dim_reducer):
    # Perform dimension reduction
    Z = dim_reducer.transform(X)
    
    # Find group
    probs = membership_vector(clusterer, Z).flatten()
    Y = np.argmax(probs) + 1
    return Y


def main(filedir, outdir=None):
    outdir = os.path.join(datadir, 'results', outdir)
    if not checkexists(outdir, size_threshold=0):
        os.makedirs(outdir)  # make output directory if not exist
        
    # Load .cif file
    cprint('Load file', filedir, '...', color='c')
    parser = CifParser(filedir)
    structure = parser.get_structures()[0]
    cprint('Done.', color='g')
    
    # Extract structural information
    cprint('Extract structural information from given .cif file.', color='c')
    ds = extract_structural_information(structure)
    cprint('Done.', color='g')
    cprint('Structure information of the queried compound is:', color='c')
    print(ds, '\n')
    
    # Apply simplification
    for stype in STYPES:
        ds['structure_' + stype] = structure_simplifications(
            ds['structure'], stype=stype, ELEMENT=ELEMENT)
    
    # Evaluate feature vector
    cprint('Evaluate feature vector...', color='c')
    ds = evaluate_feature_vector(ds, outdir=outdir)
    
    # Load group classification model
    modeldir = os.path.join(datadir, 'model_pacmap_pre')
    model_pacmap_pre = load(modeldir)
    filedir = os.path.join(datadir, 'clusterer.joblib')
    clusterer_global = joblib.load(filedir)
    
    # Load subgroup classification model
    modeldir = os.path.join(datadir, 'model_pacmap_pre_G5')
    model_pacmap_pre_G5 = load(modeldir)
    modeldir = os.path.join(datadir, 'clusterer_G5.joblib')
    clusterer_G5 = joblib.load(modeldir)
    
    modeldir = os.path.join(datadir, 'model_pacmap_pre_G7')
    model_pacmap_pre_G7 = load(modeldir)
    modeldir = os.path.join(datadir, 'clusterer_G7.joblib')
    clusterer_G7 = joblib.load(modeldir)
    
    # Load global mapping model
    modeldir = os.path.join(datadir, 'model_pacmap_initial')
    map_global = load(modeldir)
    
    # Load local mapping model
    modeldir = os.path.join(datadir, 'model_pacmap_subdivision')
    map_local = load(modeldir)
    
    # Extract data part
    X = np.vstack(ds['feature_vector']).reshape(1,-1)
    
    # Perform classification
    Y = classification(X, clusterer_global, model_pacmap_pre)
    ds['Group'] = Y
    if Y in [5, 7]:
        cprint('Classification using HDBSCAN, the material falls in Group', str(Y) + '.', 'Perform subgroup classification...', color='c')
        if Y == 5:
            YS = classification(X, clusterer_G5, model_pacmap_pre_G5)
        elif Y == 7:
            YS = classification(X, clusterer_G7, model_pacmap_pre_G7)
        subgroup = chr(ord('@') + YS)
        ds['Subgroup'] = chr(ord('@') + YS)
        cprint('The material falls in Subgroup', str(Y) + subgroup, color='g')
    else:
        cprint('The material falls in Group', str(Y) + '.', color='g')
        
    # Perform dimension reduction
    cprint('Finding coordinates in material map...', color='c')
    Z_global = map_global.transform(X).flatten()
    if Y in [5, 7]:
        Z_local = map_local.transform(X).flatten()
        cprint('Coordinates in global map:', Z_global, '| in local (subgroup) map:', Z_local, color='g')
    else:
        cprint('Coordinates in global map:', Z_global, color='g')
        
    # Overlay location on the map
    cprint('Overlay resulting coordinates in material map...', color='c')
    palette = 'cet_glasbey_category10'
    specialc = '0.6'
    specialc_loc = 'first'
    
    filedir = os.path.join(datadir, 'mp_data_clustering.pkl')
    dd = pd.read_pickle(filedir)
    arrow_locs = {y: False for y in dd['Y'].unique()}
    arrow_locs['Noise'] = None
    arrow_locs['Group 5'] = (-10, 7)
    arrow_locs['Group 7'] = (-10, 10)
    arrow_locs['Group 11'] = (-5, 5)
    arrow_locs['Group 12'] = (-10, -5)
    arrow_locs['Group 1'] = (10, -5)
    arrow_locs['Group 4'] = (-7, 7)
    group_properties = group_analysis(dd, palette=palette, specialc=specialc, 
                                        specialc_loc=specialc_loc, 
                                        simplify=True, draw=False)
    ax = plot_manifold(dd, overlay_data='conductivity', palette_data='custom_linear',
                       log_transformation=True, data_unit='log(Conductivity [S/cm])', 
                       remove_ticks=False, specialc=specialc, specialc_loc=specialc_loc, 
                       palette=palette, group_properties=group_properties, colorbar=True,
                       arrow_locs=arrow_locs, simplify=False, verbose=False)

    ax.scatter(Z_global[0], Z_global[1], c='w', marker='*', s=72, edgecolors='k', linewidths=1, zorder=100)
    filedir = os.path.join(outdir, 'global_map.png')
    plt.savefig(filedir)
    
    if Y in [5, 7]:
        cprint('Overlay resulting coordinates in local (subgroup) map...', color='c')
        filedir = os.path.join(datadir, 'mp_data_subdivision.pkl')
        dd = pd.read_pickle(filedir)
        arrow_locs = {y: False for y in dd['Y'].unique()}
        group_properties = group_analysis(dd, palette=palette, alpha=0.7, 
                                          simplify=True, draw=False)
        ax = plot_manifold(dd, overlay_data='conductivity', log_transformation=True, 
                           palette_data='custom_linear', s=6, s_data=96, fs=20,
                           data_unit='log(Conductivity [S/cm])',
                           remove_ticks=False, remove_axis=False,
                           palette=palette, mag=(10.5, 23, -4.5, 10), margin=0, 
                           group_properties=group_properties, alpha=0.7, colorbar=False,
                           arrow_locs=arrow_locs, simplify=False, verbose=False)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.subplots_adjust(bottom=0.2)
        ax.scatter(Z_local[0], Z_local[1], c='w', marker='*', s=24*10, edgecolors='k', linewidths=1.5)
        filedir = os.path.join(outdir, 'local_map.png')
        plt.savefig(filedir)
        
    # Find the nearest material in the database
    cprint('Finding structurally similar materials in the database...', color='c')
    filedir = os.path.join(datadir, 'mp_data_integrated.pkl')
    df = pd.read_pickle(filedir)
    f = ds['feature_vector']
    df['distance'] = df['descriptor'].apply(lambda x: sum((x-f)**2))
    df = df.sort_values(by='distance', ascending=True)
    dd = df[df['conductivity'].notna()]  # one with conductivity data
    attributes = ['material_id', 'formula_pretty', 'lattice_type', 'space_group', 
                  'band_gap', 'e_hull', 'conductivity', 'Z', 'Y', 'distance']
    da = dd.loc[:, attributes]
    print(da)
    plt.show()
    return


if __name__ == '__main__':
    # Parse user input
    parser = argparse.ArgumentParser(
        prog='pymatcc',
        description='Python Materials Conductivity Classifier (Pymatcc) is \
                    an open-source Python library engineered to rapidly assess \
                    the ionic conductivity potential of crystalline compounds \
                    based on their lattice structure. The library currently \
                    supports the materials containing Na (sodium) and .cif \
                    input file.',
    )

    parser.add_argument('-f', '--file', type=str, nargs=1, required=True,
                        help='File directory of the query material in .cif')
    parser.add_argument('-o', '--out', type=str, nargs=1, required=True,
                        help='Folder NAME for saving output files. The results \
                            are saved in ROOT/dat/Na/results/NAME folder.')
    args = parser.parse_args()
    filedir = args.file
    outdir = args.out
        
    if not filedir:
        cprint('File directory is not specified. Terminate the evaluation process.', color='r')
        raise()
    elif not checkexists(filedir):
        cprint('Cannot reach to the specified file directory. Terminate the evaluation process.', color='r')
        raise()
    
    # Run inspection
    main(filedir, outdir=outdir)