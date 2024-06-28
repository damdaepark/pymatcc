from pymatgen.core.periodic_table import Specie
from config import ELEMENT

def structure_simplifications(structure_input, stype, ELEMENT):
    """
    The function takes a structure and then simplifies it based on 
    the contents of simplification_dict. The original function can be found in 
    https://github.com/FALL-ML/materials-discovery.

    Parameters
    ----------
    structure_input : pymatgen.core.structure
        a pymatgen structure file

    Returns
    ------
    structure: pymatgen.core.structure
        the simplified structure
    """
    
    # Copy the structure in case modification fails
    structure = structure_input.copy()
    
    # Dictionaries for simplification
    if stype == 'A':
        simplification_dict = {'C':False, 'A':True, 'M':False, 'N':False, '40':False}
    elif stype == 'AM':
        simplification_dict = {'C':False, 'A':True, 'M':True, 'N':False, '40':False}
    elif stype == 'CAN':
        simplification_dict = {'C':True, 'A':True, 'M':False, 'N':True, '40':False}
    elif stype == 'CAMN':
        simplification_dict = {'C':True, 'A':True, 'M':True, 'N':True, '40':False}
    elif stype == 'A40':
        simplification_dict = {'C':False, 'A':True, 'M':False, 'N':False, '40':True}
    elif stype == 'AM40':
        simplification_dict = {'C':False, 'A':True, 'M':True, 'N':False, '40':True}
    elif stype == 'CAN40':
        simplification_dict = {'C':True, 'A':True, 'M':False, 'N':True, '40':True}
    elif stype == 'CAMN40':
        simplification_dict = {'C':True, 'A':True, 'M':True, 'N':True, '40':True}
    
    # Create lists to keep track of the indices for the different atom types: cation, anion, mobile, neutral
    cation_list = []
    anion_list = []
    mobile_list = []
    neutral_list = []
    
    # Create list to keep track of which atoms will be removed
    removal_list = []
    
    # Integer to keep track of how to scale the lattice (for the representations that end in '40')
    scaling_counter = 0
    
    for idx, site in enumerate(structure):
        # Grab the element name at the site
        element = site.species.elements[0].name
        
        # Grab the charge at the site
        charge = site.specie.oxi_state
        
        # If the site is the mobile atom
        if element == ELEMENT:
            mobile_list.append(idx)
        else:
            # If the site holds a neutral atom
            if charge == 0:
                neutral_list.append(idx)
                scaling_counter += 1
                structure.replace(idx, Specie('Mg', oxidation_state=charge))
            # If the site holds a cation
            elif charge > 0:
                cation_list.append(idx)
                structure.replace(idx, Specie('Al', oxidation_state=charge))
            # If the site holds an anion
            else:
                anion_list.append(idx)
                scaling_counter += 1
                structure.replace(idx, Specie('S', oxidation_state=charge))
    
    # Comparison to simplification_dict to decide which sites are removed
    if not simplification_dict['C']:
        removal_list += cation_list     
    if not simplification_dict['A']:
        removal_list += anion_list                
    if not simplification_dict['M']:
        removal_list += mobile_list
    if not simplification_dict['N']:
        removal_list += neutral_list
    
    # Special cases for the structures_A and structures_CAN representations
    # Some structures have only Na. For these we are going to handle them as anions (because every representations includes anions)
    if len(structure) == len(mobile_list):
        if not simplification_dict['M']:
            for idx in mobile_list:
                structure.replace(idx, Specie('S', oxidation_state=charge))
    
    # Some structures have only neutrals or cations. For these we are going to handle them as anions (because every representation includes anions)
    elif len(structure) == len(removal_list):
        if len(neutral_list) > 0:
            for idx in neutral_list:
                structure.replace(idx, Specie('S', oxidation_state=charge))
            structure.remove_sites(cation_list + mobile_list)
        elif len(mobile_list) > 0:
            for idx in mobile_list:
                structure.replace(idx, Specie('S', oxidation_state=charge))
            structure.remove_sites(cation_list + neutral_list)
        elif len(cation_list) > 0:
            for idx in cation_list:
                structure.replace(idx, Specie('S', oxidation_state=charge))
            structure.remove_sites(neutral_list + mobile_list)
    
    # Otherwise just remove whatever is in the removal list
    else:
        structure.remove_sites(removal_list)
    
    # If simplification_dict indicates that the lattice should be scaled
    if simplification_dict['40']:              
        if scaling_counter > 0:
            structure.scale_lattice(40*scaling_counter)
    
    return structure