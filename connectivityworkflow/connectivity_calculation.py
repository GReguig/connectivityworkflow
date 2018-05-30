#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 11:33:52 2018

@author: ghiles.reguig
"""

import numpy as np
from nipype.interfaces.base.core import LibraryBaseInterface,SimpleInterface
from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec
import nipype.interfaces.base.traits_extension as trait
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
import pandas as pd
import matplotlib.pyplot as plt
import os
plt.switch_backend("agg")

"""
Class for Nilearn Interface
"""
class NilearnBaseInterface(LibraryBaseInterface):
    _pkg = 'nilearn'
    
    
class ConnectivityCalculationInputSpec(BaseInterfaceInputSpec):
    time_series = trait.traits.Array(mandatory=True, 
                             desc=".tsv file containing a time serie for each RoI")
    kind = trait.traits.Enum('correlation', 'partial correlation', 'tangent', 'covariance', 'precision', usedefault="correlation",mandatory=False,  
                     desc="Measure of connectivity to compute, must be one of {'correlation', 'partial_correlation', 'tangent', 'covariance', 'precision'}. By default, correlation is used.")
    output_dir = trait.Str(mandatory=True, usedefault=".",
                                 desc="Directory to store generated file")
    absolute = trait.traits.Bool(mandatory=False, usedefault=False, 
                                 desc="Whether to use the absolute value of the connectivity measure. By default, False.")
    labels = trait.traits.List(mandatory=False, usedefault=None,
                               desc="List of labels associated with each time_serie")
    plotName = trait.traits.Str(mandatory=False, usedefault=None,
                                desc="Title of the connectivity matrix")
    
class ConnectivityCalculationOutputSpec(TraitedSpec):
    connectivityMatrix = trait.traits.Array(desc="Matrix of connectivity computed")
    
class ConnectivityCalculation(NilearnBaseInterface, SimpleInterface):
    
    input_spec = ConnectivityCalculationInputSpec
    output_spec = ConnectivityCalculationOutputSpec
    
    def _run_interface(self, runtime):
        self._check_kind()
        connKind = self.inputs.kind
        title = self.inputs.plotName+" "+connKind if self.inputs.plotName else  connKind
        time_series = self.inputs.time_series
        plotpath = os.path.join(self.inputs.output_dir, title)
        connMatrixPath = os.path.join(self.inputs.output_dir, title)
        labels = list(self.inputs.labels) if self.inputs.labels else None
        print("Starting connectivity calculation...")
        conn_measure = ConnectivityMeasure(kind=connKind)
        #Compute connectivity matrix
        conn_matrix = conn_measure.fit_transform([time_series])[0]
        if self.inputs.absolute : 
            conn_matrix = np.absolute(conn_matrix)
        #Set diagonal to 0
        np.fill_diagonal(conn_matrix, 0)
        print("Connectivity matrix computed.\nPlotting...")
        plt.figure()
        plotting.plot_matrix(conn_matrix, colorbar=True, labels=labels, title=title, figure=(10,8))
        plt.savefig(plotpath)
        connDataFrame = pd.DataFrame(conn_matrix, columns = labels, index=labels)
        connDataFrame.to_csv(connMatrixPath, sep="\t")
        self._results["connectivityMatrix"] = conn_matrix
        print("Connectivity calculation successfully finished")
        return runtime
                    
    def _check_kind(self):
        self.inputs.kind = self.inputs.kind.lower()
        kind = self.inputs.kind            
        print("{} connectivity matrix will be computed".format(kind))
                
    
    
    
########################## T E S T ################
"""
pathTimeSeries = time_series
output_dir = "/home/ghiles.reguig/ConnectivityWorkflow/tests/"
connCalc = ConnectivityCalculation()
#connCalc.inputs.kind="covaFFDGDnce"
connCalc.inputs.time_series = pathTimeSeries
#connCalcti.inputs.labels = labels
connCalc.inputs.absolute = True
c = connCalc.run()
"""
