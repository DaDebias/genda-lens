""" 
Utility functions for creating the dictionaries with name augmentations used to run the NER task.

Builds on early versions of the codebase developed for the following publication: 

Title: “Detecting intersectionality in NER models: A data-driven approach.”
Authors: Lassen, I. M., Almasi, M., Enevoldsen, K., & Kristensen-mclachlan, R. 
Date: 2023
Code availability: https://github.com/centre-for-humanities-computing/Danish-NER-bias
"""
import augmenty
from .process_names import (f_name_dict, m_name_dict, muslim_f_dict,
                            muslim_m_dict)

# define pattern of augmentation
patterns = [["first_name"], ["first_name", "last_name"],
            ["first_name", "last_name", "last_name"]]

# define person tag for augmenters 
person_tag = "PER" 

# define all augmenters 
f_aug = augmenty.load(
    "per_replace_v1", 
    patterns = patterns, 
    names = f_name_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )

m_aug = augmenty.load(
    "per_replace_v1", 
    patterns = patterns, 
    names = m_name_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )

muslim_f_aug = augmenty.load(
    "per_replace_v1", 
    patterns = patterns, 
    names = muslim_f_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )

muslim_m_aug = augmenty.load(
    "per_replace_v1", 
    patterns = patterns, 
    names = muslim_m_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )
