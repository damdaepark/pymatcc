from pymatgen.transformations.standard_transformations import OxidationStateDecorationTransformation
from pymatgen.transformations.standard_transformations import AutoOxiStateDecorationTransformation

oxidation_dictionary = {'H':1, 'Li':1, 'Na':1, 'K':1, 'Rb':1, 'Cs':1, 'Be':2, 'Mg':2, 'Ca':2, \
                        'Sr':2, 'Ba':2, 'Ra':2, 'B':3, 'Al':3, 'Ga':3, 'In':3, 'Tl':3, \
                        'C':4, 'Si':4, 'Ge':4, 'Sn':4, 'Pb':4, 'N':-3, 'P':5, 'As':5, \
                        'Sb':5, 'Bi':5, 'O':-2, 'S':-2, 'Se':-2, 'Te':-2, 'Po':-2, 'F':-1, \
                        'Cl':-1, 'Br':-1, 'I':-1, 'Sc':3, 'Y':3, 'Lu':3, 'Ti':4, 'Zr':4, 'Hf':4, \
                        'V':5, 'Nb':5, 'Ta':5, 'Cr':6, 'Mo':4, 'W':6, 'Mn':7, 'Tc':7, 'Re':7, \
                        'Fe':3, 'Ru':3, 'Os':3, 'Co':3, 'Rh':3, 'Ir':3, 'Cu':2, 'Ag':1, 'Au':3, \
                        'Zn':2, 'Ni':2, 'Cd':2, 'Hg':2, 'La':3, 'Ce':3, 'Pd':2, 'Pm':3, 'Ho':3, \
                        'Eu':3, 'Np':3, 'Pu':4, 'Gd':3, 'Sm':2, 'Tb':3, 'Tm':3, 'Yb':3, 'Ac':3, \
                        'Dy':3, 'Er':3, 'Pr':3, 'U':6, 'Pt':2, 'Nd':3, 'Th':4, 'Pa':5}
oxidation_decorator = OxidationStateDecorationTransformation(oxidation_dictionary)
oxidation_auto_decorator = AutoOxiStateDecorationTransformation(distance_scale_factor=1)


def apply_charge_decoration(structure, params=None, i=None):
    """
    For each structure passed into the function, the function sequentially
    attempts three charge decoration strategies:
    (1) charge decoration using record, if available (added)
    (2) manual charge decoration using the oxidation_dictionary
    (3) charge decoration using OxidationStateDecorationTransformation
    (4) charge decoration using the built-in add_oxidation_state_by_guess() method
    
    If any of these strategies result in a structure with nearly zero charge
    (<= 0.5), then the decoration is accepted.
    The original function can be found in https://github.com/FALL-ML/materials-discovery.

    Parameters
    ----------
    structure : pymatgen.core.structure
        a pymatgen structure file

    Returns
    ------
    structure: pymatgen.core.structure
        either charge decorated or not (if the decorations failed)
    """
    
    # Remove invalid charge
    structure.unset_charge()
    
    # Try the manual decoration strategy with oxidation records  #! added
    structure_copy = structure.copy()
    if hasattr(structure, 'oxi_states'):
        try:
            structure_copy.add_oxidation_state_by_element(structure.oxi_states)
            if abs(structure_copy.charge) < 0.5:
                return structure_copy
        except Exception as e:
            pass
    
    # Try the manual decoration strategy with normal oxidation number
    structure_copy = structure.copy()
    try:
        manually_transformed_structure = oxidation_decorator.apply_transformation(structure_copy)
        if abs(manually_transformed_structure.charge) < 0.5:
            return manually_transformed_structure
    except Exception as e:
        pass
    
    # Try Pymatgen's auto decorator (BVAnalyzer)
    structure_copy = structure.copy()
    try:
        auto_transformed_structure = oxidation_auto_decorator.apply_transformation(structure_copy)
        if abs(auto_transformed_structure.charge) < 0.5:
            return auto_transformed_structure
    except Exception as e:
        pass
    
    # Try Pymatgen's oxidation states approximator
    structure_copy = structure.copy()
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)
        structure_copy.add_oxidation_state_by_guess()
        signal.alarm(0)
        return structure_copy
    except Exception as e:
        pass 
    
    cprint('Failed in decoration. Use original oxidation states...', color='yellow')
    return None