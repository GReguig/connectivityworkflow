#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:56:02 2018

@author: ghiles.reguig
"""
import argparse
from nipype import config
from os.path import join as opj
import json
from connectivityworkflow.workflow_builder import BuildConnectivityWorkflow

def RunConnectivityWorkflow(path, outDir,workdir, conf_file=None):
    
    if conf_file is not None : 
        res=open(conf_file,'r').read()
        conf=json.loads(res)
        config.update_config(conf)
        
    plugin = config.get("execution","plugin")
    plugin_args = config.get("execution","plugin_args")
    if plugin_args is not None : 
        plugin_args = eval(plugin_args)
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
    #Build and run the workflow
    RunConnectivityWorkflow(path, outDir, workdir, conf)
    print("ConnectivityWorkflow Done")
    
