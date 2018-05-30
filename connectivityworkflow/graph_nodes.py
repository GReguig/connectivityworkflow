#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 10:32:35 2018

@author: ghiles.reguig
"""

from nipype import Node, Function

#################### PANDAS ADJACENCY TO NETWORKX

def pandasAdj2Nx(df):
    """
    Converts a pandas adjacency dataframe to a networkx graph
    """
    
    import networkx as nx
    import pandas as pd
    dataFrame = pd.read_csv(df,index_col=0, sep="\t")
    
    matrix = dataFrame.values
    
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise Exception("A square matrix is required. Got a matrix with shape : {}".format(matrix.shape))
    
    graph = nx.convert_matrix.from_pandas_adjacency(dataFrame)
    return graph

def NodePandasAdj2Nx():
    
    node = Node(Function(function=pandasAdj2Nx, input_names=["df"], output_names=["graph"]),
                name="Pandas2Graph")
    return node
    

################ FEATURES CALCULATOR
    
def computeFeature(graph, func, nameFeature):
    import networkx
    feature = func(graph)
    useless=0
    try:
        feature = dict(feature)
    except Exception:
        useless+=1
    return {nameFeature : feature}

def NodeComputeFeature():
    node = Node(Function(function=computeFeature, input_names=["graph", "func", "nameFeature"], 
                output_names=["feature"]), name="FeatureCalculator")
    return node 

############## Join FEATURES
    
def joinFeatures(data, output_dir, prefix, confName, kindConn):
    import pandas as pd
    from os.path import join as opj
    import os
    """
    data : liste de dictionnaires. Soit mesure globale : len == 1, sinon locale : len == nbRoI
    """
    df = pd.DataFrame()
    for feature in data :
        #If local measures (1 per node)
        featureVal = feature.values()[0]
        if not isinstance(featureVal, float):
            serie = pd.Series(featureVal.values(), index=featureVal.keys())
            df[feature.keys()[0]] = serie
        else:
            df[feature.keys()[0]] = featureVal
    outdir = opj(output_dir,"GraphMeasures")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    df.to_csv(opj(outdir,prefix+confName+kindConn+"GraphFeatures.tsv"),sep="\t")
    return df 
        
def NodeJoinFeatures():
    node = Node(Function(function=joinFeatures, input_names=["data", "prefix","output_dir", "confName", "kindConn"], output_names=["graphFeatures"]), name="JoinFeatures")
    return node 

def ShowRes(data):
    print("\n\nGot : {}\n\n".format(data))

########################### T E S T #######
"""
conn = pd.read_csv("/home/ghiles.reguig/ConnectivityWorkflow/test/testConn.tsv",index_col=0, sep="\t")

#matrix, labels = conn.values[:,1:], list(conn.columns[1:])

graph = nx.convert_matrix.from_pandas_adjacency(conn)

nx.drawing.nx_agraph.write_dot(graph, "/home/ghiles.reguig/ConnectivityWorkflow/test/test")
plt.subplot(121)
nx.draw(graph, with_labels=True, font_weight="bold")
plt.subplot(122)
nx.draw_shell(graph, with_labels=True)

nodeConvert = NodePandasAdj2Nx()
nodeConvert.inputs.df = conn

nodeFeature = MapNode(Function(function=computeFeature, input_names=["graph","func","nameFeature"],
                      output_names=["feature"]), name="FeatureCalculator", iterfield=["func","nameFeature"])

nodeFeature.inputs.func = [networkx.degree, networkx.density, networkx.betweenness_centrality]
nodeFeature.inputs.nameFeature = ["degre", "densite", "bc"]


#showNode = Node(Function(function=ShowRes, input_names=["data"], output_names=[]), name="Show")
showNode = Node(Function(function=joinFeatures, input_names=["data", "output_dir"], output_names=["merged"]),name="Merge")

showNode.inputs.output_dir = "/export/dataCENIR/users/ghiles.reguig/testBIDSB0/derivatives/connectivityWorkflow/sub-10109PAR/ses-M0/func/"

wf = Workflow(name="GraphWf")

wf.connect(nodeConvert, "graph", nodeFeature, "graph")
wf.connect(nodeFeature, "feature", showNode, "data")
res = wf.run()
"""