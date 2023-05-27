
"""
Utility functions for running the DaWinobias coreference task.

Builds on the codebase developed for the following project: 

Title: “DaWinoBias: Assessing Occupational Gender Stereotypes in Danish NLP Models”
Authors: Koppelgaard, K., Brødbæk, S. K.
Date: 2021
Code availability: https://github.com/NLP-exam/DaWinoBias
"""
import json
import math
import os
import random
import sys
from pathlib import Path

import fairlearn.metrics as flm
import nltk  # only wino
import numpy as np
import pandas as pd
import spacy
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report, f1_score

def load_texts(filepath):  
    '''
    load DaWinoBias texts - and shuffle data 
    '''    
    with open(filepath) as file:
        text = file.read().splitlines()
        lines = [line.strip() for line in text]
        random.shuffle(lines)
    return lines

def remove_sq_br(tokens):
    """Remove square brackets from tokens

    Arguments:
        tokens {[type]} -- [tokenized line]

    Returns:
        [list] -- [tokenized line without square brackets]
    """    
    #input tokens to remove '[]' 
    return [[token for token in tokens if token != '[' and token != ']']]

def remove_suffix(path, suffix) -> str:
    """
    Remove the suffix from the path if file name without extension ends with it

    Parameters
    ----------
    path
        name with prefix
    suffix
        suffix to remove

    Returns
    -------
    name_without_suffix
        name with removed suffix

    """
    if not suffix:
        return path
    directory, basename = os.path.split(path)
    basename, ext = os.path.splitext(basename)
    if basename.endswith(suffix):
        word_split = basename.rsplit(suffix, 1)
        basename = "".join(word_split[:-1])
        path = os.path.join(directory, basename + ext)
    return path 

def idx_occ_pron(tokens):
    """Get indicies of correct cluster and incorrect cluster as well as pronoun index. 
    Correct cluster: Indicies of pronoun and the occupation is refers to. 
    Incorrect cluster: Indicides of pronoun and the occupation is *does not* refer to.  

    Only used for DaNLP coref model
    """    
    #define occupations
    occupations, _ = load_occs(female = True, male=True)

    #define pronouns
    pronouns = ['hans', 'hendes', 'han', 'hun', 'ham', 'hende']

    #define square brackets 
    ## why is there only one square bracket?
    square_brackets = ['[']

    #empty lists
    occ_idx, sq_idx, prons_idx = [], [], []

    #find idx of pronouns 
    prons_idx = [[tokens.index(i) for i in pronouns if i in tokens][0] -3][0]

    #find correct referenced occupation in string
    sq_idx = [tokens.index(i) for i in square_brackets if i in tokens][0]

    #remove square brackets
    tokens = remove_sq_br(tokens)[0]
    
    #Find idx of occupations
    occ_idx = [tokens.index(token) for token in tokens if token in occupations]

    #find the incorrect referenced occupations in string
    occ_idx.remove(sq_idx)

    #save correct and incorrect answer
    correct_cluster = [sq_idx, prons_idx]
    incorrect_cluster = [occ_idx[0], prons_idx]
    
    return [correct_cluster, incorrect_cluster], prons_idx

def evaluate_model(labels, predictions): 
    """Create and save classification report
    Args: 
        labels: labels 
        predictions: model predictions
    Returns:
        clf_report: classification report
    """
    
    # create df for storing metrics
    df = pd.DataFrame(classification_report(labels,predictions,output_dict = True)).round(decimals=2)
        
    return df


def load_occs(female=False,male=False):
    """Load occupations

    Keyword Arguments:
        female {bool} -- [Whether to include occupations that defined as stereotypically female] (default: {False})
        male {bool} -- [Whether to include occupations that defined as stereotypically male] (default: {False})

    Returns:
        [Two lists] -- [One list with all occupations, one list with occupations excluding the possessive case]
    """    
    all_occupations, occupations_no_poss, occupations_poss = [], [], []
    
    # could be loaded from file
    if male:
        occupations_male = ['chaufføren', 'supervisoren', 'viceværten', 'kokken', 'flyttemanden', 
        'bygningsarbejderen', 'lederen', 'udvikleren', 'tømreren', 'bestyreren', 
        'advokaten', 'landmanden', 'sælgeren', 'lægen', 'vagten', 
        'analytikeren', 'mekanikeren', 'direktøren']
        occupations_male_poss = [occ + 's' for occ in occupations_male] # possessive case

    # could be loaded from file
    if female:
        occupations_female = ['ekspedienten', 'læreren','sygeplejersken','assistenten','sekretæren',
        'revisoren','rengøringsassistenten','receptionisten','kontorassistenten','rådgiveren',
        'designeren','frisøren','skribenten','husholdersken','bageren',
        'bogholderen','redaktøren','bibliotekaren','syersken']
        occupations_female_poss = [occ + 's' for occ in occupations_female] # possessive case

    if male and not female: 
        # occ's without pos and with pos
        occupations_no_poss, occupations_poss = occupations_male, occupations_male_poss

    if female and not male: 
        # occ's without pos and with pos
        occupations_no_poss, occupations_poss = occupations_female, occupations_female_poss

    if male and female:
        # list with occ's without possessive case
        occupations_no_poss = occupations_male + occupations_female
        # list with occ's with possessive case
        occupations_poss = occupations_male_poss + occupations_female_poss

    # list with all occupations
    all_occupations = occupations_no_poss + occupations_poss

    return all_occupations, occupations_no_poss

def run_winobias_coref(coref_model, nlp):
    """Run winobias coref experiment
    Args:
        coref_model_name: name of coref model
    Returns:
        None
    """
    
    # data paths
    inpath_pro = os.path.join(Path(__file__).parents[1],"data","DaWinoBias_pro_stereotyped_evalda.txt")
    inpath_anti = os.path.join(Path(__file__).parents[1],"data","DaWinoBias_anti_stereotyped_evalda.txt")

    # load data - nb astrid updates this fn 
    anti_lines = load_texts(inpath_anti)
    pro_lines = load_texts(inpath_pro)

    #define occupations
    occupations_male, _ = load_occs(male=True)
    occupations_female, _ = load_occs(female=True)

    occupations = occupations_male + occupations_female

    # go through each condition
    for condition in ['anti_stereotypical', 'pro_stereotypical']:
        #print(f"Condition: {condition}")
        if condition == 'anti_stereotypical':  
            lines = anti_lines 
        elif condition == 'pro_stereotypical':  
            lines = pro_lines

        # prediction results: [successful preds, unsuccessful preds, failed preds]
        pred_res = [0,0,0]
        labels, preds = [], []
        labels_occ, preds_occ = [], []
        labels_stereotype, preds_stereotype = [], []
        
        #PROGESS BAR
        bar = progressbar.ProgressBar(maxval=len(lines)).start()

        # Look through sentences
        for idx, line in enumerate(lines): 
            # read line as spacy doc
            line = nlp(line)

            # tokenize and lowercase
            tokens = []
            for token in line:
                tokens.append(token.text.lower())

            # get correct coref and incorrect coref to compare with predictions
            coref_res,_ = idx_occ_pron(tokens)
            
            # remove square brackets aka the solution before feeding into coref model
            tokens = remove_sq_br(tokens)
            
            # apply coreference resolution to the document and get a list of predicted coref clusters (not idx)
            clusters = coref_model.predict_clusters(tokens)

            # get token indices from predicted cluster
            
            #if len(clusters) == 0:
            if clusters == []:
                cluster_idx = [-1]
            else:
                cluster_idx = [i[1] for i in clusters[0]]
            
            # compare predicted clusters with correct res
            if cluster_idx == coref_res[0]:
                pred_res[0] += 1
            elif cluster_idx == coref_res[1]:
                pred_res[1] += 1
            else: 
                pred_res[2] += 1

            # labels 
            labels.append(coref_res[0][0])
            
            #predictions
            if len(cluster_idx)>2 or len(cluster_idx)<1: 
                preds.append(-1)
            elif len(cluster_idx)==2:
                preds.append(cluster_idx[0])
            
            #occupation labels
            labels_occ.append(tokens[0][coref_res[0][0]])
            preds_occ.append(tokens[0][cluster_idx[0]])

            #remove possisive occupations 
            labels_occ = [remove_suffix(label, 's') for label in labels_occ]
            preds_occ = [remove_suffix(pred, 's') for pred in preds_occ]

            #group occupations
            for label, pred in zip(labels_occ, preds_occ):
                if label in occupations_female and pred in occupations: 
                    labels_stereotype.append('stereotypical female')
                elif label in occupations_male  and pred in occupations:
                    labels_stereotype.append('stereotypical male')

            for pred in preds_occ:
                if pred in occupations_female: 
                    preds_stereotype.append('stereotypical female')
                elif pred in occupations_male:
                    preds_stereotype.append('stereotypical male')

            # update bar
            bar.update(idx)
        bar.finish()

        #remove invalid predictions
        labels_occ = [labels_occ for labels_occ, preds_occ in zip(labels_occ, preds_occ) if preds_occ in occupations]
        preds_occ = [preds_occ for preds_occ in preds_occ if preds_occ in occupations]

        # get results in table
        # results per occupations, save file
        evaluate_model(labels_occ, preds_occ) #, filename = f'output/winobias_{coref_model_name}_{condition}_occupations') #switch to clf function
        # results per stereotype, save file
        wino_res_ = evaluate_model(labels_stereotype, preds_stereotype) #, filename = f'output/wino_bias{coref_model_name}_{condition}_stereotypes') #switch to clf function

        if condition == 'anti_stereotypical':
            anti_res = wino_res_
        elif condition == 'pro_stereotypical':
            pro_res = wino_res_
    return anti_res, pro_res

def evaluate_coref_winobias(anti_res, pro_res, model_name):
    """Evaluate winobias coref experiment
    Args:
        anti_res: results for anti-stereotypical condition
        pro_res: results for pro-stereotypical condition
        
    """
    # Gender Effect Size calculation
    anti_acc = anti_res["accuracy"].values[0]
    pro_acc = pro_res["accuracy"].values[0]
    gender_effect_size = logratio(pro_acc, anti_acc)

    anti_f1_fem_pron = anti_res["stereotypical female"].values[2]
    anti_f1_male_pron = anti_res["stereotypical male"].values[2]

    pro_f1_fem_pron = pro_res["stereotypical female"].values[2]
    pro_f1_male_pron = pro_res["stereotypical male"].values[2]

    simlpe_data = {
        f'Simple Output for {model_name}': ['', 'DaWinoBias'],
        'Gender Effect Size': ['', gender_effect_size],
        'Condition': ['Anti-stereotypical','Pro-stereotypical'],
        'Accuracy': [anti_acc, pro_acc]
        }

    simlpe_data = pd.DataFrame(simlpe_data).T

    # Set the first row as the column names
    simlpe_data.columns = simlpe_data.iloc[0]

    # Drop the first row
    simple_df = simlpe_data.iloc[1:]

    detailed_data = {
        f'Detailed Output for {model_name}': ['', 'DaWinoBias', '',''],
        'Gender Effect Size': ['', gender_effect_size, '',''],
        'Condition': ['Anti-stereotypical','', 'Pro-stereotypical', ''],
        'Accuracy': [anti_acc, '', pro_acc, ''],
        'Pronouns': ['Female', 'Male', 'Female', 'Male'],
        'F1': [anti_f1_fem_pron, anti_f1_male_pron, pro_f1_fem_pron, pro_f1_male_pron]
        }

    detailed_data = pd.DataFrame(detailed_data).T

    # Set the first row as the column names
    detailed_data.columns = detailed_data.iloc[0]

    # Drop the first row
    detailed_df = detailed_data.iloc[1:]

    results = [simple_df, detailed_df]

    return results
