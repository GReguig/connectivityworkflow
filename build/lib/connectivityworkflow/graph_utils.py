"""

This module contains related graph functions that can be used as functions in 
nipype nodes.

"""

def pandas_to_nx(df):
    """
    Converts a pandas adjacency dataframe to a networkx graph.
    """
    
    import networkx as nx
    import pandas as pd
    
    dataFrame = pd.read_csv(df,index_col=0, sep="\t")
    
    graph = nx.convert_matrix.from_pandas_adjacency(dataFrame)
    return graph


def computeFeature(graph, func, nameFeature):
    """
    Wrapper to compute a feature function on a graph and name it. 
    """
    import networkx
    feature = func(graph)
    useless=0
    try:
        feature = dict(feature)
    except Exception:
        useless+=1
    return {nameFeature : feature}


def computeHubness(graph, outdir):
    import networkx as nx 
    hubness_matrix = nx.algorithms.hub_matrix(graph)
    

################### T  E S T ############
"""

pathConn = ("/export/dataCENIR/users/ghiles.reguig/testBIDSB0/derivatives/"
            "connectivityWorkflow/sub-10110GEF/ses-M0/func/"
            "connectivityMeasures/Aroma-precision/"
            "sub-10110GEF_ses-M0_task-rest-preproc_Aroma-precision.tsv")

g = pandas_to_nx(pathConn)
"""