import datetime
import os
import random

from dreamcoder.dreamcoder import explorationCompression
from dreamcoder.program import *
from dreamcoder.grammar import Grammar
from dreamcoder.domains.list.experiments.runUtils import *
from dreamcoder.properties.utilsPlotting import *

def main(args):
    """
    Takes the return value of the `arguments()` function as input and
    trains/tests the model on manipulating sequences of numbers.
    """

    # extractorName = args.pop("extractor")    
    # propSamplingGrammarWeights = args.pop("propSamplingGrammarWeights")
    # save = args.pop("save")
    libraryName = args.pop("libraryName")
    dataset = args.pop("dataset")
    # debug = args.pop("debug")
    # hidden = args.pop("hidden")
    # propDreamTasks = args.pop("propDreamTasks")
    # propNumIters = args.pop("propNumIters")
    # hmfSeed = args.pop("hmfSeed")
    # propSamplingPrimitives = args.pop("propSamplingPrimitives")


    tasks = get_tasks(dataset)
    print("Loaded {} tasks".format(len(tasks)))
    prims = get_primitives(libraryName)
    baseGrammar = Grammar.uniform([p for p in prims])

    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/jrule/%s/"%timestamp
    os.system("mkdir -p %s"%outputDirectory)
    
    args.update({
        "outputPrefix": "jrule",
        "outputDirectory": outputDirectory,
        "evaluationTimeout": 0.0005,
    })

    random.seed(args["seed"])

    explorationCompression(baseGrammar, tasks, testingTasks=[], **args)