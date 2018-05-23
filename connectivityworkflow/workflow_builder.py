from connectivityworkflow.data_bids_grabber import GetBidsDataGrabberNode
from connectivityworkflow.confounds_selector import getConfoundsReaderNode
from connectivityworkflow.signal_extraction_freesurfer import SignalExtractionFreeSurfer
from connectivityworkflow.connectivity_calculation import ConnectivityCalculation
from nipype import Workflow, Node

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
