#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 10:25:41 2018

@author: ghiles.reguig
"""

from sklearn.base import BaseEstimator, TransformerMixin
from nibabel.freesurfer import read_annot
import numpy as np
from nipype import Node, Function
#from nilearn.surface import load_surf_data

def extract_hemisphere_time_series(surf_data, mask_array):
    """
    Function for time-series extraction from a mask array
    """
    roi_indices = np.unique(mask_array)[1:]
    masks_RoI = [mask_array == roi_num for roi_num in roi_indices]
    time_series= [[np.mean(frame*mask) for frame in surf_data.T] 
                                            for mask in masks_RoI]
    return np.asarray(time_series)


class SurfaceMasker(BaseEstimator, TransformerMixin):
    
    """
    Class for time-series extraction from surface data
    """
    
    def __init__(self, annot_file):
        self.mask, _, self.names = read_annot(annot_file)
    
    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X):
        return extract_hemisphere_time_series(X, self.mask)

    def get_names(self):
        return self.names


def extract_time_series(lh_surf, rh_surf, lh_annot, rh_annot):
    
    import numpy as np
    from connectivityworkflow.surface_processing import extract_hemisphere_time_series
    from nilearn.surface import load_surf_data
    
    lh_mask, _, lh_names = read_annot(lh_annot)
    rh_mask, _, rh_names = read_annot(rh_annot)
    
    lh_surf_data = load_surf_data(lh_surf)
    rh_surf_data = load_surf_data(rh_surf)
    
    time_series_lh = extract_hemisphere_time_series(lh_surf_data, lh_mask)
    time_series_rh = extract_hemisphere_time_series(rh_surf_data, rh_mask)
    
    return np.concatenate((time_series_lh, time_series_rh)).T, lh_names
    

def get_time_series_extractor_node():
    
    return Node(Function(function=extract_time_series,
                         input_names=["lh_surf","rh_surf", "lh_annot","rh_annot"],
                         output_names=["time_series", "roi_names"]), name="SurfaceTimeSeriesExtractor")

####################### T E S T #############
""" 
gordonLAnnot = ("/export/dataCENIR/users/ghiles.reguig/AtlasGordon/"
                "Gordon333_FSannot/lh.Gordon333.annot")
gordonRAnnot = ("/export/dataCENIR/users/ghiles.reguig/AtlasGordon/"
                "Gordon333_FSannot/rh.Gordon333.annot")

pBoldL = ("/export/dataCENIR/users/ghiles.reguig/testBIDSB0/derivatives/"
          "fmriprep/sub-10109PAR/ses-M0/func/"
          "sub-10109PAR_ses-M0_task-rest_bold_space-fsaverage.L.func.gii")


pBoldR = ("/export/dataCENIR/users/ghiles.reguig/testBIDSB0/derivatives/"
          "fmriprep/sub-10109PAR/ses-M0/func/"
          "sub-10109PAR_ses-M0_task-rest_bold_space-fsaverage.R.func.gii")

boldFuncL = load_surf_data(pBoldL)
boldFuncR = load_surf_data(pBoldR)
   
surf_masker_l = SurfaceMasker(gordonLAnnot)
surf_masker_r = SurfaceMasker(gordonRAnnot)

surf_data_roi = np.concatenate((surf_masker_l.fit_transform(boldFuncL), 
                                surf_masker_r.fit_transform(boldFuncR)))
"""