
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.domains.list.main import main, list_options
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.properties.utils import property_options
from dreamcoder.properties.utilsPropertySampling import prop_sampling_options
from dreamcoder.properties.propertySignatureExtractor import PropertySignatureExtractor
from dreamcoder.type import arrow, tlist, tint, tbool
from dreamcoder.utilities import numberOfCPUs

def extras(parser):
    for f in [property_options, prop_sampling_options, list_options]:
        parser = f(parser)

if __name__ == '__main__':
    args = commandlineArguments(
        enumerationTimeout=10, activation='tanh', iterations=10, recognitionTimeout=3600,
        a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
        helmholtzRatio=1.0, structurePenalty=1., useRecognitionModel=True,
        CPUs=numberOfCPUs(), featureExtractor=PropertySignatureExtractor,
        extras=extras
    )
    args["propertyRequest"] = arrow(tlist(tint), tlist(tint), tbool)
    main(args)
