#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 10:14:32 2018

@author: ghiles.reguig
"""

import numpy as np 

def to_symmetric_matrix(arrays, distance):
    
    """
    Computes a symmetric distance/similarity matrix from a set of
    arrays.
    
    Parameters
    ----------
    arrays : ndarray, shape depends on the distance function input
        (n_samples, distance_input_shape)
        
    distance : a distance/similarity function taking as input 2 elements of
        arrays and returning a value
        
    Returns
    -------
    distance_matrix : A symmetric matrix containing the distances/similarities
        between arrays' objects, shape (n_samples, n_samples)
    
    """
    #Compute upper_triangular_values of distance matrix
    upper_triangular_values = [[distance(line, column) for column in arrays[index:]]
        for index, line in enumerate(arrays)]
    #Matrix where 
    distance_matrix = np.empty((len(arrays), len(arrays)))
    for i, vals in enumerate(upper_triangular_values):
        distance_matrix[i, i:] = vals
    i_upper = np.triu_indices(len(arrays), 0)
    distance_matrix.T[i_upper] = distance_matrix[i_upper]
    
    return distance_matrix#upper_triangular_values#distance_matrix*distance_matrix.T


def rv_coef(mat1, mat2):
    trace_m1m2 = np.trace(mat1.T.dot(mat2))
    trace_m1 = np.trace(mat1.T.dot(mat1))
    trace_m2 = np.trace(mat2.T.dot(mat2))
    return trace_m1m2/np.sqrt(trace_m1*trace_m2)