
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

# from dreamcoder.domains.list.taskProperties import handWritteProperties, handWrittenPropertyFuncs, tinput, toutput
from dreamcoder.likelihoodModel import UniqueTaskSignatureScore, TaskDiscriminationScore
from dreamcoder.domains.list.listPrimitives import test_josh_rich_primitives
from dreamcoder.domains.list.main import list_options
from dreamcoder.domains.list.propSimMain import main
from dreamcoder.domains.list.property import Property
from dreamcoder.domains.list.propertySignatureExtractor import testPropertySignatureExtractorHandwritten
from dreamcoder.domains.list.resultsProcessing import viewResults
from dreamcoder.domains.list.sampleProperties import prop_sampling_options
from dreamcoder.domains.list import utilsPlotting
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs


if __name__ == '__main__':
    # args = commandlineArguments(
    #     enumerationTimeout=10, activation='tanh', iterations=10, recognitionTimeout=3600,
    #     a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
    #     helmholtzRatio=1.0, structurePenalty=1., useRecognitionModel=True,
    #     CPUs=numberOfCPUs(),
    #     extras=lambda parser: list_options(prop_sampling_options(parser)),
    #     )
    # main(args)
    utilsPlotting.main()
