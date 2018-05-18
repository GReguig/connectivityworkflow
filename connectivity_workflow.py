#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:56:02 2018

@author: ghiles.reguig
"""
import argparse
from .data_bids_grabber import GetBidsDataGrabberNode
from .confounds_selector import getConfoundsReaderNode
from .signal_extraction_freesurfer import SignalExtractionFreeSurfer
from .connectivity_calculation import ConnectivityCalculation
from nipype import Workflow, Node
from os.path import join as opj

def BuildConnectivityWorkflow(path, outDir):
    #Workflow Initialization
    connectivityWorkflow = Workflow(name="connectivityWorkflow")
    #Input Node for reading BIDS Data
    inputNode = GetBidsDataGrabberNode(path)
    inputNode.inputs.outDir = outDir
    #Confound selector
    confoundsReader = getConfoundsReaderNode()
    confoundsReader.iterables = [('regex', [("[^(Cosine|aCompCor|tCompCor|AROMAAggrComp)\d+]", "minimalConf"),
                                              ("[^(Cosine|tCompCor|AROMAAggrComp)\d+]","aCompCor"),
                                              ("[^(Cosine|aCompCor|AROMAAggrComp)\d+]", "tCompCor"),
                                              ("[^(Cosine|aCompCor|tCompCor)\d+]", "Aroma")])]
    #Signal Extraction
    signalExtractor = Node(SignalExtractionFreeSurfer(), name="SignalExtractor")
    #Connectivity Calculation
    connectivityCalculator = Node(ConnectivityCalculation(), name="ConnectivityCalculator")
    connectivityCalculator.iterables = [("kind", ["correlation", "covariance", "precision", "partial correlation"])]
    connectivityCalculator.inputs.absolute = True
    #Workflow connections
    connectivityWorkflow.connect([
            (inputNode, confoundsReader, [("confounds","filepath")]),
            (inputNode, signalExtractor, [("aparcaseg","roi_file"),
                                          ("preproc", "fmri_file"),
                                          ("outputDir", "output_dir")]),
            (confoundsReader, signalExtractor, [("values","confounds"),
                                                ("confName","confoundsName")]),
            (signalExtractor, connectivityCalculator, [("time_series","time_series"),
                                                       ("roiLabels", "labels"),
                                                       ("confName", "plotName")]),
            (inputNode, connectivityCalculator, [("outputDir", "output_dir")])
            ])
    return connectivityWorkflow

def RunConnectivityWorkflow(path, outDir,workdir):
    wf = BuildConnectivityWorkflow(path,outDir)
    wf.base_dir = workdir
    wf.run(plugin='SLURMGraph', plugin_args = {'dont_resubmit_completed_jobs': True,
"sbatch_args":" -e /export/dataCENIR/users/ghiles.reguig/ConnOutput/%j.err -o /export/dataCENIR/users/ghiles.reguig/ConnOutput/%j.out --mem 12G"})
    
    
    
if __name__ == "__main__" : 
    #Argument parser
    parser = argparse.ArgumentParser(description="Path to BIDS Dataset")
    #Add the filepath argument to the BIDS dataset
    parser.add_argument("-p","--path", dest="path",help="Path to BIDS Dataset", required=True)
    parser.add_argument("-w", "--workdir", dest="workdir", help="Path to working directory", required=False)
    #Parse the commandline
    args = parser.parse_args()
    #Get the filepath argument specified by the user
    path = args.path
    workdir = args.workdir
    outDir = opj(path,"derivatives","connectivityWorkflow")
    
    print("Results will be written in {}".format(outDir))
    #Build the workflow
    RunConnectivityWorkflow(path, outDir, workdir)
    