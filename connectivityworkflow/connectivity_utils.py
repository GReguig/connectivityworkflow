#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 18:01:23 2018

@author: ghiles.reguig
"""


from nipype import Node, Function

from nilearn import surface
from surface_processing import extract_time_series

def p_val_mask(series, thr=0.01):
    import numpy as np
    from scipy.stats import pearsonr
    
    pearson_res = np.asarray([[pearsonr(series[:,i], series[:,j]) 
        for i in range(series.shape[1])] 
            for j in range(series.shape[1])])
    p_values = pearson_res[:,:,1]
    mask = p_values < 0.01
    return mask


def get_p_val_mask_node():
    return Node(Function(function = p_val_mask, 
                                      input_names=["series"],
                                      output_names=["mask"]),
                                      name="PValueConnectivityMask")

#################### T E S T ############
    
pBoldL = "/export/dataCENIR/users/ghiles.reguig/testBIDSB0/derivatives/fmriprep/sub-10110GEF/ses-M0/func/sub-10110GEF_ses-M0_task-rest_bold_space-fsaverage.L.func.gii"
pBoldR = "/export/dataCENIR/users/ghiles.reguig/testBIDSB0/derivatives/fmriprep/sub-10110GEF/ses-M0/func/sub-10110GEF_ses-M0_task-rest_bold_space-fsaverage.R.func.gii"
#Rea
boldFuncL = surface.load_surf_data(pBoldL)
boldFuncR = surface.load_surf_data(pBoldR)
gordonLAnnot = "/export/dataCENIR/users/ghiles.reguig/AtlasGordon/Gordon333_FSannot/lh.Gordon333.annot"
gordonRAnnot = "/export/dataCENIR/users/ghiles.reguig/AtlasGordon/Gordon333_FSannot/rh.Gordon333.annot"

time_series, names = extract_time_series(boldFuncL, boldFuncR, gordonLAnnot, gordonRAnnot)

mask=p_val_mask(time_series)
