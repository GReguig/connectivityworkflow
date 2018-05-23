#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:56:02 2018

@author: ghiles.reguig
"""
import argparse
from connectivityworkflow.data_bids_grabber import GetBidsDataGrabberNode
from connectivityworkflow.confounds_selector import getConfoundsReaderNode
from connectivityworkflow.signal_extraction_freesurfer import SignalExtractionFreeSurfer
from connectivityworkflow.connectivity_calculation import ConnectivityCalculation
from nipype import Workflow, Node, config
from os.path import join as opj
import json

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

def RunConnectivityWorkflow(path, outDir,workdir, conf_file=None):
    if conf_file is not None : 
        res=open(conf_file,'r').read()
        conf=json.loads(res)
        config.update_config(conf)
        
    plugin = config.get("execution","plugin")
    plugin_args = eval(config.get("execution","plugin_args"))
    wf = BuildConnectivityWorkflow(path,outDir)
    wf.base_dir = workdir
    wf.run(plugin=plugin, plugin_args=plugin_args)
    
    
    
    
if __name__ == "__main__" : 
    #Argument parser
    parser = argparse.ArgumentParser(description="Path to BIDS Dataset")
    #Add the filepath argument to the BIDS dataset
    parser.add_argument("-p","--path", dest="path",help="Path to BIDS Dataset", required=True)
    parser.add_argument("-w", "--workdir", dest="workdir", help="Path to working directory", required=False)
    parser.add_argument("-c", "--config", dest="conf", help="Nipype JSON configuration file ", required=False)
    #Parse the commandline
    args = parser.parse_args()
    #Get the filepath argument specified by the user
    path = args.path
    workdir = args.workdir
    conf = args.conf
    outDir = opj(path,"derivatives","connectivityWorkflow")
    
    print("Results will be written in {}".format(outDir))
    #Build the workflow
    RunConnectivityWorkflow(path, outDir, workdir, conf)
    
