from connectivityworkflow.data_bids_grabber import GetBidsDataGrabberNode
from connectivityworkflow.confounds_selector import getConfoundsReaderNode
from connectivityworkflow.signal_extraction_freesurfer import SignalExtractionFreeSurfer
from connectivityworkflow.connectivity_calculation import ConnectivityCalculation
from nipype import Workflow, Node, MapNode
import networkx
from connectivityworkflow.graph_nodes import NodePandasAdj2Nx, computeFeature, Function, NodeJoinFeatures

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
                                          ("outputDir", "output_dir"),
                                          ("prefix", "prefix")]),
            (confoundsReader, signalExtractor, [("values","confounds"),
                                                ("confName","confoundsName")]),
            (signalExtractor, connectivityCalculator, [("time_series","time_series"),
                                                       ("roiLabels", "labels"),
                                                       ("confName", "plotName")]),
            (inputNode, connectivityCalculator, [("outputDir", "output_dir")])
            ])
    ## GRAPH

    pandas2Graph = NodePandasAdj2Nx()
    #GraphFeature Calculator
    graphFeature = MapNode(Function(function=computeFeature, input_names=["graph","func","nameFeature"],
                      output_names=["feature"]), name="FeatureCalculator", iterfield=["func","nameFeature"])
    
    graphFeature.inputs.func = [networkx.degree, networkx.density, networkx.betweenness_centrality, networkx.algorithms.clustering, networkx.algorithms.transitivity, networkx.average_shortest_path_length]
    graphFeature.inputs.nameFeature = ["Degree", "Density", "Betweenness-Centrality", "Average-Clustering", "Global-Clustering", "Average-Shortest-Path-Length"]
    #Join Features
    joinFeatures = NodeJoinFeatures()
    
    connectivityWorkflow.connect([
            (pandas2Graph, graphFeature, [("graph","graph")]),
            (graphFeature, joinFeatures, [("feature","data")]),
            (inputNode, joinFeatures, [("outputDir","output_dir"), 
                                       ("prefix","prefix")]),
            (connectivityCalculator, pandas2Graph, [("dfPath", "df")]),
            (confoundsReader, joinFeatures, [("confName","confName")]),
            (connectivityCalculator, joinFeatures, [("kind", "kindConn")])
            ])

    return connectivityWorkflow

