""" 
Utility functions for creating the names lists used to run the NER task.

Builds on early versions of the codebase developed for the following publication: 

Title: “Detecting intersectionality in NER models: A data-driven approach.”
Authors: Lassen, I. M., Almasi, M., Enevoldsen, K., & Kristensen-mclachlan, R. 
Date: 2023
Code availability: https://github.com/centre-for-humanities-computing/Danish-NER-bias
"""
import os
from pathlib import Path

import pandas as pd
from dacy.datasets import load_names, muslim_names

def remove_duplicates(all_names, names_to_filter_away):
    all_names = [name for name in all_names if name not in names_to_filter_away]
    return all_names

# path to names files 
data_folder = os.path.join(Path(__file__).parents[1], "data","name_aug_csv_files") 

### Define majority/danish names

# get last names 
last_names_2023 = pd.read_csv(os.path.join(data_folder, "last_names_2023.csv"))
last_names_2023["Navn"] = last_names_2023["Navn"].str.title() # capitalize
last_names_2023 = list(last_names_2023["Navn"])[:500] # subset to only 500 to match 500 first names

# men and women first names
men_2023 = pd.read_csv(os.path.join(data_folder, "first_names_2023_men.csv"))
women_2023 = pd.read_csv(os.path.join(data_folder, "first_names_2023_women.csv"))

# capitalize
men_2023["Navn"] = men_2023["Navn"].str.title()
women_2023["Navn"] = women_2023["Navn"].str.title()

# subset names to 500 
men_2023 = list(men_2023["Navn"])[:500]
women_2023 = list(women_2023["Navn"])[:500]

# create dictionaries 
m_name_dict = {'first_name':men_2023, 'last_name':last_names_2023}
f_name_dict = {'first_name':women_2023, 'last_name':last_names_2023}

### Define muslim/minority names 
muslim_name_dict = muslim_names()
muslim_m_dict = load_names(ethnicity="muslim", gender="male", min_prop_gender=0.5)
muslim_f_dict = load_names(ethnicity="muslim", gender="female", min_prop_gender=0.5)

### Remove overlaps 
# read in annotated
overlaps = pd.read_csv(os.path.join(data_folder, "overlapping_names.csv"))

# create muslim/minority only list and majority/danish only list 
muslim_only=list(overlaps.query("origin=='M'")["duplicates"])
danish_only=list(overlaps.query("origin=='DK'")["duplicates"])

# majority/danish seperate genders into seperate dicts
f_name_dict["first_name"] = remove_duplicates(f_name_dict["first_name"], muslim_only)
m_name_dict["first_name"] = remove_duplicates(m_name_dict["first_name"], muslim_only)

# muslim/minority seperate genders into seperate dicts
muslim_f_dict["first_name"] = remove_duplicates(muslim_f_dict["first_name"], danish_only)
muslim_m_dict["first_name"] = remove_duplicates(muslim_m_dict["first_name"], danish_only)
