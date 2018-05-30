# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import nilearn
from nilearn.input_data import NiftiLabelsMasker
import nipype.interfaces.base.traits_extension as trait
from nipype.interfaces.base.core import LibraryBaseInterface,SimpleInterface
from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec
import os
import pandas as pd

"""
Class for Nilearn Interface
"""
class NilearnBaseInterface(LibraryBaseInterface):
    _pkg = 'nilearn'


"""
Class for RoI signal extraction
"""
class SignalExtractionFreeSurferInputSpec(BaseInterfaceInputSpec):
    #fMRI data as a nifti File
    fmri_file = trait.ImageFile(exists=True, mandatory=True, 
                              desc="4-D fMRI nii file")
    #FreeSurfer's RoI File
    roi_file = trait.ImageFile(exists=True, mandatory=True, 
                               desc="A FreeSurfer's segmentation file (aparcaseg file, for example)")
    confounds = trait.traits.Array(exists=True, mandatory=False, usedefault=None, 
                           desc="Array containing fMRI confounds for signal cleaning")
    confoundsName = trait.traits.Str(mandatory=False, usedefault=None, desc="Name of the set of counfounds to use for signal cleaning")
    lutFile = trait.File(exists=True, mandatory=False, usedefault=None,
                         desc="FreeSurfer's Color Looking Up Table")
    output_dir = trait.Directory(exists=False, mandatory=False, usedefault=".")
    prefix = trait.traits.Str(mandatory=False, usedefault="", desc="Prefix of the bids files")
        
class SignalExtractionFreeSurferOutputSpec(TraitedSpec):
    time_series = trait.traits.Array(desc="Array containing time series/RoI. Dimensions : (timestamps x RoI)")
    roiLabels = trait.traits.List(desc="List of labels associated with each RoI")
    confName = trait.traits.Str(desc="Name of the set of confounds used for signal cleaning")

class SignalExtractionFreeSurfer(NilearnBaseInterface, SimpleInterface):
    
    '''
    
    '''
    input_spec = SignalExtractionFreeSurferInputSpec
    output_spec = SignalExtractionFreeSurferOutputSpec
    
    def _run_interface(self, runtime):
        print("\nExtracting signal from FreeSurfer's RoI...\n")
        #Getting the confounds
        if isinstance(self.inputs.confounds, np.ndarray) :
            confounds = self.inputs.confounds  
            confName = self.inputs.confoundsName
        else:
            confounds = None
            confName = "NoConfounds"
        if not self.inputs.lutFile : 
            self.inputs.lutFile = os.path.join(os.environ['FREESURFER_HOME'],"FreeSurferColorLUT.txt")
        #Mask to extract time_series/RoI        
        masker = NiftiLabelsMasker(self.inputs.roi_file)
        #Time series extracted/RoI, dimensions (timestamps x nb RoI)
        roiTimeSeries = masker.fit_transform(self.inputs.fmri_file, confounds=confounds)
        #Getting LUT Table
        lutTable = np.loadtxt(self.inputs.lutFile, dtype=str)[1:]
        #RoIs present in the roi_file
        rois_Present  = np.unique(nilearn.image.load_img(self.inputs.roi_file).get_data())
        #Names of the RoIs
        rois_Present_Names = [region[1] for region in lutTable if int(region[0]) in rois_Present]
        #DataFrame containing time_series/RoI
        time_seriesDF = pd.DataFrame(roiTimeSeries, columns=rois_Present_Names)
        #Name of the OutPutFile
        directory = os.path.join(self.inputs.output_dir,"time_series")
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except Exception:
                print("exception makedirs")
        outFile = os.path.join(directory, self.inputs.prefix+self.inputs.confoundsName+"TimeSeriesRoI.tsv")
        #Output value
        self._results["time_series"] = time_seriesDF.values
        self._results["roiLabels"] = list(time_seriesDF.columns)
        self._results["confName"] = confName
        print("Time series successfully computed.\nSaving data in {}\n\n".format(outFile))
        #Saving as a tsv file
        time_seriesDF.to_csv(outFile, sep="\t", index=False)
        return runtime
    
            
        
    
#################################### T E S T #######################
"""
aparcaseg = "/export/dataCENIR/users/ghiles.reguig/fmriprep/processedAroma/fmriprep/sub-20102GAI/ses-M0/func/sub-20102GAI_ses-M0_task-rest_bold_space-T1w_label-aparcaseg_roi.nii.gz"
#Serie fonctionnelle dans l'espace T1w
boldFunc = "/export/dataCENIR/users/ghiles.reguig/fmriprep/processedAroma/fmriprep/sub-20102GAI/ses-M0/func/sub-20102GAI_ses-M0_task-rest_bold_space-T1w_preproc.nii.gz"
#LUT 
lookUpTable = "/export/data/opt/CENIR/freesurfer6.0_cento06/FreeSurferColorLUT.txt"
#Confounds File
conf = "/export/dataCENIR/users/ghiles.reguig/fmriprep/processedAroma/fmriprep/sub-20102GAI/ses-M0/func/sub-20102GAI_ses-M0_task-rest_bold_confounds.tsv"

extractFS = SignalExtractionFreeSurfer()
extractFS.inputs.fmri_file = boldFunc
extractFS.inputs.roi_file = aparcaseg
extractFS.inputs.output_dir = "/home/ghiles.reguig/ConnectivityWorkflow/tests/"
#extractFS.inputs.lutFile = lookUpTable
extractFS.inputs.confounds = conf
res = extractFS.run()
"""