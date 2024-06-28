import os
import pickle
import joblib

import numpy as np
import pandas as pd
import matminer.featurizers.composition as mm_composition  # pip install matminer
import matminer.featurizers.structure as mm_structure
from dscribe.descriptors import SOAP  # conda install -c conda-forge dscribe
from pymatgen.io import ase

from utils import datadir, imgdir, cprint, flatten_list, checkexists, isstring, \
    isnumeric, istuple, isscalar, isarray, isstringdata
from config import ELEMENT, STRUCTURES
datadir = os.path.join(datadir, ELEMENT)


def calculate_unique_atoms(structures):
    """
    Calculate the unqiue atoms in the structure. Used for the SOAP representation. 
    """
    unique_atoms = []
    for structure in structures:
        try:
            symbol_set = structure.symbol_set
        except:  # only one element
            symbol_set = [structure.specie.element.symbol]
            
        for num in symbol_set:
            if num not in unique_atoms:
                unique_atoms.append(num)
    return np.sort(unique_atoms)


def get_featurizer(featurizer, structures, compositions, tmode=None, etype=None, 
                   modeldir=None):
    if etype == 'structure':
        if tmode == 'train':
            featurizer.fit(structures)
            with open(modeldir, 'wb') as f:
                pickle.dump(featurizer, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(modeldir, 'rb') as f:
                featurizer = pickle.load(f)
    elif etype == 'composition':
        if tmode == 'train':
            featurizer.fit(compositions)
            with open(modeldir, 'wb') as f:
                pickle.dump(featurizer, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(modeldir, 'rb') as f:
                featurizer = pickle.load(f)
    else:
        raise('Unexpected behavior!')
    return featurizer


def eval_featurizer(featurizer, structures, compositions, etype=None):    
    cprint('Evaluate', featurizer.__class__, '...', color='c')
    result = []
    if etype == 'structure':
        for i, structure in enumerate(structures):
            try:
                result.append(featurizer.featurize(structure))
            except:
                result.append(None)
    elif etype == 'composition':
        for i, composition in enumerate(compositions):
            try:
                result.append(featurizer.featurize(composition))
            except:
                result.append(None)
    else:
        raise('Unexpected behavior!')
    return result


def get_feature_vector(df, rlist):
    feature_vector = []
    for (pair, data) in df.items():
        if isscalar(data):
            feature_vector.append(data)
        elif isarray(data):
            _data = list(np.array(data)[rlist[pair][1]])  # select the valid elements screened in training
            feature_vector.extend(_data)
        else:
            raise('#TODO')
    return feature_vector


def eval_descriptor(simplification, descriptor, ds, tmode, outdir, 
                    save_results=True, parallel=True):
    pair = '(' + simplification + ', ' + descriptor + ')'
    
    if descriptor == 'gii':
        params = (20,)  # (rcut)
    elif descriptor == 'rdf':
        params = (10, 0.1)  # (cutoff, bin_size)
    elif descriptor == 'SOAP':
        params = ('outer', 3, 5, 3)  # (average, rcut, nmax, lmax)
    elif descriptor == 'xrd':
        params = (451,)  # (pattern_length)
    else:
        params = None
    
    structure = ds[simplification]
    if descriptor in ['ape', 'bc', 'end', 'md', 'os', 'vo', 'yss']:
        composition = structure.composition
    else:  # no need to evaluate compositions
        composition = None
    structures = [structure]
    compositions = [composition]
    unique_atoms = calculate_unique_atoms(structures)
    
    index, modeldir = indexer(descriptor, simplification, params)
    tempdir = os.path.join(outdir, 'descriptors', index + '.pkl')
    if checkexists(tempdir):
        with open(tempdir, 'rb') as f:
            result = pickle.load(f)
    else:
        cprint('Evaluating descriptor vectors for', pair, 'pair...', color='w')
        if descriptor in mapper.keys():
            func = mapper[descriptor]
            result = func(structures, compositions, tmode=tmode, modeldir=modeldir)
        elif descriptor == 'gii':
            result = run_global_instability_index_featurizer(
                structures, compositions, rcut=params[0], tmode=tmode, 
                modeldir=modeldir)
        elif descriptor == 'rdf':
            result = run_rdf_featurizer(
                structures, compositions, cutoff=params[0], bin_size=params[1], 
                tmode=tmode, modeldir=modeldir)
        elif descriptor == 'SOAP':
            result = run_SOAP(structures, unique_atoms=unique_atoms, 
                                rcut=params[1], nmax=params[2], lmax=params[3], 
                                average=params[0], tmode=tmode, 
                                modeldir=modeldir)
        elif descriptor == 'xrd':
            result = run_XRD_featurizer(structures, compositions, 
                                        pattern_length=params[0], 
                                        tmode=tmode, modeldir=modeldir, 
                                        parallel=parallel)
        else:
            raise('#TODO')
        if save_results:
            with open(tempdir, 'wb') as f:
                pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
                    
    return result


def replace_nan(x):
    x = np.array(x)
    x[np.isnan(x)] = 0
    return list(x)


def evaluate_feature_vector(ds, outdir, tmode='test'):
    filedir = os.path.join(datadir, 'header.pkl')
    with open(filedir, 'rb') as f:
        header = pickle.load(f)
        
    filedir = os.path.join(datadir, 'rlist.pkl')
    with open(filedir, 'rb') as f:
        rlist = pickle.load(f)
        
    # Evaluate/Load descriptor values
    pairs = [pair for pair in header.keys() if '@' in pair]
    for pair in pairs:
        descriptor, simplification = pair.split('@')
        ds[pair] = eval_descriptor(simplification, descriptor, ds, 
                                   tmode=tmode, outdir=outdir, 
                                   save_results=True, parallel=True)
    
    # Convert tuple to list
    cprint('Converting tuple to list...', color='w')
    for (column, data) in ds.items():
        if istuple(data):
            ds[column] = list(data)
    
    # Unpack scalar list
    cprint('Unpacking scalar list...', color='w')
    for (column, data) in ds.items():
        if isscalar(data):
            ds[column] = data[0] if isinstance(data, list) else data
    
    # Treat nan values
    cprint('Treating nan values...', color='w')
    for (column, data) in ds.items():
        if isarray(data) and not isstringdata(data):
            ds[column] = replace_nan(data)
        elif data == np.nan:
            ds[column] = 0
    
    # Convert label data to metric
    cprint('Integer encoding of the structure-relevant properties...', color='w')
    filedir = os.path.join(datadir, 'encoding.pkl')
    with open(filedir, 'rb') as f:
        _dict_space_group = pickle.load(f)
        dict_space_group = _dict_space_group[0]
        dict_lattice_type = _dict_space_group[1]
    ds['space_group_encoded'] = dict_space_group[ds['space_group']]
    ds['lattice_type_encoded'] = dict_lattice_type[ds['lattice_type']]
    
    # Evaluate volume and Na element composition ratio
    cprint('Evaluate additional descriptors...', color='w')
    ds['V'] = ds['volume']
    
    elements = ds['elements']
    stoichiometry = ds['stoichiometry']
    comp_Na = stoichiometry[elements.index(ELEMENT)]
    comp_total = sum(stoichiometry)
    ds[ELEMENT + '_ratio'] = comp_Na/comp_total
    
    # Check the length of each descriptor and match the order of the descriptors
    cprint('Check if the descriptor lengths match with those in training dataset...', color='w')
    _ds = ds.iloc[:22]
    ds_ = ds.iloc[22:]
    ds_list = []
    for key in header.keys():
        if key in ['space_group_encoded', 'lattice_type_encoded', 'V', ELEMENT + '_ratio']:
            continue
        values = ds_[key]
        if isarray(values):
            assert len(values) == len(rlist[key][0])  # array length of descriptor should match with training dataset
            ds_list.extend(values)
        else:
            ds_list.append(values)
    
    # Generate scaled feature vector
    cprint('Generate feature vector...', color='w')
    feature_vector = get_feature_vector(ds_, rlist=rlist)
    filedir = os.path.join(datadir, 'scaler.joblib')
    scaler = joblib.load(filedir)
    ds['feature_vector'] = scaler.transform(np.array(feature_vector).reshape(1,-1)).flatten()
    return ds


def run_atomic_packing_efficiency_featurizer(structures, compositions, tmode=None, modeldir=None, parallel=False):
    featurizer = mm_composition.AtomicPackingEfficiency()
    featurizer = get_featurizer(featurizer, structures, compositions, tmode, etype='composition', modeldir=modeldir)
    result = eval_featurizer(featurizer, structures, compositions, etype='composition')
    return result
    
    
def run_band_center_featurizer(structures, compositions, tmode=None, modeldir=None, parallel=False):
    featurizer = mm_composition.BandCenter()
    featurizer = get_featurizer(featurizer, structures, compositions, tmode, etype='composition', modeldir=modeldir)
    result = eval_featurizer(featurizer, structures, compositions, etype='composition')
    return result


def run_bond_fraction_featurizer(structures, compositions, tmode=None, modeldir=None, parallel=False):
    featurizer = mm_structure.BondFractions()
    featurizer = get_featurizer(featurizer, structures, compositions, tmode, etype='structure', modeldir=modeldir)
    result = eval_featurizer(featurizer, structures, compositions, etype='structure')
    return result

    
def run_chemical_ordering_featurizer(structures, compositions, tmode=None, modeldir=None, parallel=False):
    featurizer = mm_structure.ChemicalOrdering()
    featurizer = get_featurizer(featurizer, structures, compositions, tmode, etype='structure', modeldir=modeldir)
    result = eval_featurizer(featurizer, structures, compositions, etype='structure')
    return result


def run_density_featurizer(structures, compositions, tmode=None, modeldir=None, parallel=False):
    featurizer = mm_structure.DensityFeatures(("density", "vpa", "packing fraction"))
    featurizer = get_featurizer(featurizer, structures, compositions, tmode, etype='structure', modeldir=modeldir)
    result = eval_featurizer(featurizer, structures, compositions, etype='structure')
    return result


def run_electronegativity_difference_featurizer(structures, compositions, tmode=None, modeldir=None, parallel=False):
    featurizer = mm_composition.ElectronegativityDiff()
    featurizer = get_featurizer(featurizer, structures, compositions, tmode, etype='composition', modeldir=modeldir)
    result = eval_featurizer(featurizer, structures, compositions, etype='composition')
    return result


def run_ewald_energy_featurizer(structures, compositions, tmode=None, modeldir=None, parallel=False):
    featurizer = mm_structure.EwaldEnergy()
    featurizer = get_featurizer(featurizer, structures, compositions, tmode, etype='structure', modeldir=modeldir)
    result = eval_featurizer(featurizer, structures, compositions, etype='structure')
    return result


def run_global_instability_index_featurizer(structures, compositions, rcut, tmode=None, modeldir=None, parallel=False):
    featurizer = mm_structure.GlobalInstabilityIndex(r_cut=rcut)
    featurizer = get_featurizer(featurizer, structures, compositions, tmode, etype='structure', modeldir=modeldir)
    result = eval_featurizer(featurizer, structures, compositions, etype='structure')
    if len(result) == 0:
        return None
    else:
        return flatten_list(result)  # flatten for the case when [0.0]
    

def run_jarvis_cfid_featurizer(structures, compositions, tmode=None, modeldir=None, parallel=False):
    featurizer = mm_structure.JarvisCFID()
    featurizer = get_featurizer(featurizer, structures, compositions, tmode, etype='structure', modeldir=modeldir)
    result = eval_featurizer(featurizer, structures, compositions, etype='structure')
    return result


def run_maximum_packing_efficiency_featurizer(structures, compositions, tmode=None, modeldir=None, parallel=False):
    featurizer = mm_structure.MaximumPackingEfficiency()
    featurizer = get_featurizer(featurizer, structures, compositions, tmode, etype='structure', modeldir=modeldir)
    result = eval_featurizer(featurizer, structures, compositions, etype='structure')
    return result


def run_meredig_featurizer(structures, compositions, tmode=None, modeldir=None, parallel=False):
    featurizer = mm_composition.Meredig()
    featurizer = get_featurizer(featurizer, structures, compositions, tmode, etype='composition', modeldir=modeldir)
    result = eval_featurizer(featurizer, structures, compositions, etype='composition')
    return result

    
def run_orbital_field_matrix_featurizer(structures, compositions, tmode=None, modeldir=None, parallel=False):
    featurizer = mm_structure.OrbitalFieldMatrix(period_tag=True)
    featurizer = get_featurizer(featurizer, structures, compositions, tmode, etype='structure', modeldir=modeldir)
    result = eval_featurizer(featurizer, structures, compositions, etype='structure')
    return result

    
def run_oxidation_states_featurizer(structures, compositions, tmode=None, modeldir=None, parallel=False):
    featurizer = mm_composition.OxidationStates()
    featurizer = get_featurizer(featurizer, structures, compositions, tmode, etype='composition', modeldir=modeldir)
    result = eval_featurizer(featurizer, structures, compositions, etype='composition')
    return result

    
def run_rdf_featurizer(structures, compositions, cutoff, bin_size, tmode=None, modeldir=None, parallel=False):
    featurizer = mm_structure.RadialDistributionFunction(cutoff=cutoff, bin_size=bin_size)
    featurizer = get_featurizer(featurizer, structures, compositions, tmode, etype='structure', modeldir=modeldir)
    result = eval_featurizer(featurizer, structures, compositions, etype='structure')
    return result
                     

def run_sine_coulomb_featurizer(structures, compositions, tmode=None, modeldir=None, parallel=False):
    featurizer = mm_structure.SineCoulombMatrix()
    featurizer = get_featurizer(featurizer, structures, compositions, tmode, etype='structure', modeldir=modeldir)
    result = eval_featurizer(featurizer, structures, compositions, etype='structure')
    return result
            

AAA = ase.AseAtomsAdaptor
def run_SOAP(structures, unique_atoms, rcut, nmax, lmax, average, tmode=None, modeldir=None, parallel=False):
    if len(unique_atoms) == 1:
        result = [None]*len(structures)
    else:
        result = []
        for structure in structures:
            average_soap = SOAP(species=unique_atoms, r_cut=rcut,
                                n_max=nmax, l_max=lmax, periodic=True,
                                average=average, sparse=True)
            atoms = AAA.get_atoms(structure)
            average_soap_data = average_soap.create(atoms, n_jobs=n_jobs, verbose=True)
            
            # Mobile-to-mobile ions
            try:
                pair = ('Na', 'Na')
                indices = np.r_[average_soap.get_location(pair)]
            except:
                try:
                    pair = ('Al', 'Al')
                    indices = np.r_[average_soap.get_location(pair)]
                except:
                    pair = ('S', 'S')
                    indices = np.r_[average_soap.get_location(pair)]
            result.append(average_soap_data[indices].todense())
        return result
    

def run_structural_complexity_featurizer(structures, compositions, tmode=None, modeldir=None, parallel=False):
    featurizer = mm_structure.StructuralComplexity()
    featurizer = get_featurizer(featurizer, structures, compositions, tmode, etype='structure', modeldir=modeldir)
    result = eval_featurizer(featurizer, structures, compositions, etype='structure')
    return result

                
def run_structural_heterogeneity_featurizer(structures, compositions, tmode=None, modeldir=None, parallel=False):
    featurizer = mm_structure.StructuralHeterogeneity()
    featurizer = get_featurizer(featurizer, structures, compositions, tmode, etype='structure', modeldir=modeldir)
    result = eval_featurizer(featurizer, structures, compositions, etype='structure')
    return result


def run_valence_orbital_featurizer(structures, compositions, tmode=None, modeldir=None, parallel=False):
    featurizer = mm_composition.ValenceOrbital()
    featurizer = get_featurizer(featurizer, structures, compositions, tmode, etype='composition', modeldir=modeldir)
    result = eval_featurizer(featurizer, structures, compositions, etype='composition')
    return result
    

def run_XRD_featurizer(structures, compositions, pattern_length, tmode=None, modeldir=None, parallel=False):
    featurizer = mm_structure.XRDPowderPattern(pattern_length=pattern_length)
    featurizer = get_featurizer(featurizer, structures, compositions, tmode, etype='structure', modeldir=modeldir)
    result = eval_featurizer(featurizer, structures, compositions, etype='structure')
    return result


def run_yang_solid_solution_featurizer(structures, compositions, tmode=None, modeldir=None, parallel=False):
    featurizer = mm_composition.YangSolidSolution()
    featurizer = get_featurizer(featurizer, structures, compositions, tmode, etype='composition', modeldir=modeldir)
    result = eval_featurizer(featurizer, structures, compositions, etype='composition')
    return result


mapper = {
    'ape': run_atomic_packing_efficiency_featurizer,
    'bc': run_band_center_featurizer,
    'bf': run_bond_fraction_featurizer,
    'co': run_chemical_ordering_featurizer,
    'density': run_density_featurizer,
    'end': run_electronegativity_difference_featurizer,
    'ee': run_ewald_energy_featurizer,
    'jcfid': run_jarvis_cfid_featurizer,
    'mpe': run_maximum_packing_efficiency_featurizer,
    'md': run_meredig_featurizer,
    'ofm': run_orbital_field_matrix_featurizer,
    'os': run_oxidation_states_featurizer,
    'scm': run_sine_coulomb_featurizer,
    'sc': run_structural_complexity_featurizer,
    'sh': run_structural_heterogeneity_featurizer,
    'vo': run_valence_orbital_featurizer,
    'yss': run_yang_solid_solution_featurizer
}


def indexer(descriptor, simplification, params=None):
    if descriptor == 'gii':
        index = 'gii_rcut-{}_mode-{}'.format(params[0], simplification)
        modeldir = os.path.join(datadir, 'models', index + '.pkl')
    elif descriptor == 'rdf':
        index = 'rdf_cutoff-{}_binsize-{}_mode-{}'.format(params[0], params[1], simplification)
        modeldir = os.path.join(datadir, 'models', index + '.pkl')
    elif descriptor == 'SOAP':
        index = 'SOAP_partialS_{}_rcut-{}_nmax-{}_lmax-{}_mode-{}'.format(params[0], params[1], params[2], params[3], simplification)
        modeldir = os.path.join(datadir, 'models', index + '.pkl')
    elif descriptor == 'xrd':
        index = 'xrd_pattern_length-{}_mode-{}'.format(params[0], simplification)
        modeldir = os.path.join(datadir, 'models', index + '.pkl')
    else:
        index = descriptor + '_mode-' + simplification
        modeldir = os.path.join(datadir, 'models', index + '.pkl')
    return index, modeldir
