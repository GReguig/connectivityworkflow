#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 12:56:23 2018

@author: ghiles.reguig
"""

from nipype import Node, Function


def confoundsReader(filepath, regex=None):
    import pandas as pd
    #Reading the confounds tsv file
    confounds = pd.read_csv(filepath, sep="\t")
    if regex : 
        reg, confName = regex
        confounds = confounds.filter(regex=reg)
    #Replacing missing values by 0.0
    confounds = confounds.fillna(0.0)
    return confounds.values, confounds.columns, confName
        

def getConfoundsReaderNode():
    
    confReader = Node(Function(function=confoundsReader, input_names=["filepath","regex"],                                
                               output_names=["values", "headers","confName"]), name="ConfoundsReader")
    return confReader

