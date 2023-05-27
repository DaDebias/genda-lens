""" 
Utility functions for running the NER task.

Builds on early versions of the codebase developed for the following publication: 

Title: “Detecting intersectionality in NER models: A data-driven approach.”
Authors: Lassen, I. M., Almasi, M., Enevoldsen, K., & Kristensen-mclachlan, R. 
Date: 2023
Code availability: https://github.com/centre-for-humanities-computing/Danish-NER-bias
"""

import math
import os

import pandas as pd
import spacy
from dacy.score import score

def load_mdl(model_name):
    """Loading pre-trained model using DaCy or from Hugging Face via spacy_wrap 

    Args:
        model_name (str): name of model in DaCy or on Hugging Face 

    Raises:
        ValueError: raise error if HF model cannot be loaded from Hugging Face
        ValueError: raise error if model type not supported in package 

    Returns:
        model: NER model  
    """    
    import dacy
    if model_name in dacy.models():
        print(f"[INFO] Loading model {model_name} with DaCy.")
        import dacy

        #model = dacy.load("da_dacy_large_trf-0.2.0")
        model = dacy.load(model_name)
    elif model_name not in dacy.models():
        print(f"[INFO] Loading model {model_name} from Hugging Face with spaCy-wrap.")
        try:
            #import spacy
            import spacy_wrap
            model = spacy.blank("da")
            # config for loading 
            config = {"model": {"name": model_name}, 
                        "predictions_to": ["ents"]} 
            model.add_pipe("token_classification_transformer", config=config)
        except OSError:
            raise ValueError("Cannot load model from Hugging Face. Please specify the full name of the model i.e. 'name/model-name' ")     
    else:
        raise ValueError("Cannot load model. This package only supports models from DaCy and Hugging Face.")
    return model

def eval_model_augmentation(mdl, model_name, n, augmenters, dataset):
    """Compute NER performance on the DaNe test set using DaCy's score function

    Args:
        mdl (model): NER model  
        model_name (str): model name
        n (int): number of repetitions to run 
        augmenters (generator): dictionary with name augmentaions to test 
        dataset (corpus): the DaNe test set 

    Returns:
        list: list with two df's: output condensed and output detailed  
    """    
    # loop over all models in model_dict 
    scores = []
    i = 0
    for aug, nam, k in augmenters:
        print(f"\t [INFO] Running augmenter: {nam} with {k} repetitions.")
        scores_ = score(corpus=dataset, apply_fn=mdl, augmenters=aug, k=k)
        scores_["model"] = model_name
        scores_["augmenter"] = nam
        scores_["i"] = i
        scores.append(scores_)
        i += 1
    scores_df = pd.concat(scores)

    # summarise scores over augmentations, compute gender effect
    scores_summed, l_ratio = get_means_sds(scores_df)

    # compute simple and detailed output
    out_detailed = eval_ner_detailed(scores_summed, l_ratio, model_name)

    out_simple = eval_ner_simple(scores_summed, l_ratio, model_name)

    return [out_simple, out_detailed]

def logratio(x:float ,y:float):
    return round(math.log2(x/ y), 2)

def get_means_sds(df):
    '''
    Compute mean F1 and SD for each augmenter of a single model.

    input:
     - df with model performance of a single model. 

    output: 
    -  
    '''
    print(df.head())
    df["f1"] = df["ents_excl_MISC_ents_f"] #*100
    df = df[df["augmenter"].isin(["Majority female names", "Majority male names", "Minority male names", "Minority female names"])]
    augs, means, sds = [],[],[]
    for aug in set(df["augmenter"]):
        augs.append(aug)
        sub = df[df["augmenter"]==aug]
        means.append(round(sub["f1"].mean(),4))
        sds.append(round(sub["f1"].std(),2))
    # create df
    out_df = pd.DataFrame({"Group": augs, "F1": means, "SD": sds})
    # create df with mean and SD in one column
    combinations = [str(str(a)+" ("+str(b)+")") for (a,b) in zip(means,sds)]
    out_df_condensed = pd.DataFrame({"Group": augs, "F1 (SD)": combinations})

    # Compute log-ratio
    f = out_df[out_df["Group"]=="Majority female names"].iloc[0,1]
    m = out_df[out_df["Group"]=="Majority male names"].iloc[0,1]
    ratio = logratio(m,f)  

    return out_df_condensed, ratio


def eval_ner_simple(df, l_ratio, model_name):
    data = {
        f'Simple Output {model_name}': ['', 'Augmented DaNe', ''],
        'Gender Effect Size': ['', l_ratio,''],
        'Name Augmentation': ['Female', '', 'Male'],
        'Macro F1 (SD)': [df[df["Group"]=="Majority female names"].iloc[0,1], '', 
                     df[df["Group"]=="Majority male names"].iloc[0,1]],
    }
    df_out = pd.DataFrame(data).T
    df_out.columns = df_out.iloc[0]
    df_out = df_out.iloc[1:]
    return df_out

def eval_ner_detailed(df, l_ratio, model_name):
    data = {
        f'Detailed Output for {model_name}': ['', 'Augmented DaNe', '', ''],
        'Gender Effect Size': ['', l_ratio,'', ''],
        'Name Augmentation': ['Female', '', 'Male', ''],
        'Macro F1 (SD)': [df[df["Group"]=="Majority female names"].iloc[0,1],'', df[df["Group"]=="Majority male names"].iloc[0,1],''],
        'Name Augmentation ': ['Majority Female', 'Minority Female', 'Majority Male', 'Minority Male'],
        'Macro F1 (SD) ': [df[df["Group"]=="Majority female names"].iloc[0,1], 
                      df[df["Group"]=="Minority female names"].iloc[0,1], 
                      df[df["Group"]=="Majority male names"].iloc[0,1], 
                      df[df["Group"]=="Minority male names"].iloc[0,1]]
    }
    out_df = pd.DataFrame(data).T
    out_df.columns = out_df.iloc[0]
    out_df = out_df.iloc[1:]
    return out_df

