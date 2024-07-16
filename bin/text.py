try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.type import arrow, tlist, tcharacter, tbool

from dreamcoder.domains.text.main import main, text_options
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs

from dreamcoder.properties.propertySignatureExtractor import PropertySignatureExtractor
from dreamcoder.properties.utils import property_options
from dreamcoder.properties.utilsPropertySampling import prop_sampling_options


if __name__ == '__main__':

    def extras(parser):
        for f in [property_options, prop_sampling_options, text_options]:
            parser = f(parser)

    arguments = commandlineArguments(
        recognitionTimeout=7200,
        iterations=10,
        helmholtzRatio=0.5,
        topK=2,
        maximumFrontier=5,
        structurePenalty=10.,
        a=3,
        activation="tanh",
        CPUs=numberOfCPUs(),
        featureExtractor=PropertySignatureExtractor,
        pseudoCounts=30.0,
        extras=extras)

    arguments["propertyRequest"] = arrow(tlist(tcharacter), tlist(tcharacter), tbool)
    main(arguments)
