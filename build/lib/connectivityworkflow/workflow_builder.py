from connectivityworkflow.data_bids_grabber import GetBidsDataGrabberNode, get_bids_surf_data_node
from connectivityworkflow.confounds_selector import getConfoundsReaderNode
from connectivityworkflow.signal_extraction_freesurfer import SignalExtractionFreeSurfer
from connectivityworkflow.connectivity_calculation import ConnectivityCalculation
from connectivityworkflow.surface_processing import get_time_series_extractor_node
from connectivityworkflow.connectivity_utils import get_p_val_mask_node
from nipype import Workflow, Node, MapNode, JoinNode
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
            (inputNode, connectivityCalculator, [("outputDir", "output_dir"),
                                                 ("prefix","prefix")])
            ])
    ## GRAPH

    pandas2Graph = NodePandasAdj2Nx()
    #GraphFeature Calculator
    graphFeature = MapNode(Function(function=computeFeature, input_names=["graph","func","nameFeature"],
                      output_names=["feature"]), name="FeatureCalculator", iterfield=["func","nameFeature"])
    
    graphFeature.inputs.func = [networkx.clustering, networkx.algorithms.local_efficiency, 
                                networkx.algorithms.global_efficiency, networkx.degree, 
                                networkx.algorithms.centrality.betweenness_centrality]
    graphFeature.inputs.nameFeature = ["clustering", "local_efficiency", 
                                       "global_efficiency", "degree", 
                                       "betweenness_centrality"]
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


def BuildConnectivityWorkflowSurface(path, outDir, lh_annot="/export/dataCENIR/users/ghiles.reguig/AtlasGordon/Gordon333_FSannot/lh.Gordon333.annot", 
                                     rh_annot="/export/dataCENIR/users/ghiles.reguig/AtlasGordon/Gordon333_FSannot/rh.Gordon333.annot"):
    #Workflow Initialization
    connectivityWorkflow = Workflow(name="connectivityWorkflow")
    #Input Node for reading BIDS Data
    inputNode = get_bids_surf_data_node(path)
    inputNode.inputs.outDir = outDir
    #Confound selector
    confoundsReader = getConfoundsReaderNode()
    confoundsReader.iterables = [('regex', [("[^(Cosine|aCompCor|tCompCor|AROMAAggrComp)\d+]", "minimalConf"),
                                            ("[^(Cosine|aCompCor|tCompCor|AROMAAggrComp|GlobalSignal)\d+]","minimalConfNoGlobal"),
                                              ("[^(Cosine|tCompCor|AROMAAggrComp)\d+]","aCompCor"),
                                              ("[^(Cosine|aCompCor|AROMAAggrComp)\d+]", "tCompCor"),
                                              ("[^(Cosine|aCompCor|tCompCor)\d+]", "Aroma")])]
    #Signal Extraction
    signalExtractor = get_time_series_extractor_node()
    signalExtractor.inputs.lh_annot = lh_annot
    signalExtractor.inputs.rh_annot = rh_annot
    #Connectivity Calculation
    connectivityCalculator = Node(ConnectivityCalculation(), name="ConnectivityCalculator")
    connectivityCalculator.iterables = [("kind", ["correlation", "covariance", "precision", "partial correlation"])]
    connectivityCalculator.inputs.absolute = True
    """
    tangentCalculator = JoinNode(ConnectivityCalculation(), name="TangentCalculator", joinsource="SurfaceTimeSeriesExtractor", joinfield="time_series")
    tangentCalculator.inputs.kind="tangent"
    """
    #Workflow connections
    connectivityWorkflow.connect([
            (inputNode, confoundsReader, [("confounds","filepath")]),
            (inputNode, signalExtractor, [("lh_surf","lh_surf"),
                                          ("rh_surf", "rh_surf"),
                                          ("outputDir", "output_dir"),
                                          ("prefix", "prefix")]),
            (confoundsReader, signalExtractor, [("values","confounds"),
                                                ("confName","confoundsName")]),
            (signalExtractor, connectivityCalculator, [("time_series","time_series"),
                                                       ("roiLabels", "labels"),
                                                       ("confName", "plotName")]),
            (inputNode, connectivityCalculator, [("outputDir", "output_dir"),
                                                 ("prefix", "prefix")])
            ])
    ## GRAPH
    """
    pandas2Graph = NodePandasAdj2Nx()
    #GraphFeature Calculator
    graphFeature = MapNode(Function(function=computeFeature, input_names=["graph","func","nameFeature"],
                      output_names=["feature"]), name="FeatureCalculator", iterfield=["func","nameFeature"])
    
    graphFeature.inputs.func = [networkx.clustering, networkx.degree, 
                                networkx.algorithms.centrality.betweenness_centrality]
    graphFeature.inputs.nameFeature = ["clustering", "degree", 
                                       "betweenness_centrality"]
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
    """
    return connectivityWorkflow

############################# T E S T ###########################
"""
path = "/export/dataCENIR/users/ghiles.reguig/testBIDSB0"
wf = BuildConnectivityWorkflowSurface(path, ".")

wf.run(plugin='MultiProc', plugin_args={'n_procs' : 28})


"""
