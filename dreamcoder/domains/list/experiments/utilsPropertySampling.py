import dill

from dreamcoder.domains.list.handwrittenProperties import tinput, toutput, handWrittenProperties
from dreamcoder.domains.list.listPrimitives import josh_primitives, bootstrapTarget_extra
from dreamcoder.domains.list.makeListTasks import joshTasks, sortBootstrap

from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.properties.utilsPropertySampling import setPropertyGrammarWeights
from dreamcoder.type import *

OUTPUT_STR = "$0"
MIN_LOG_PRIOR = -11
DATA_DIR = "data/prop_sig/"
SAMPLED_PROPERTIES_DIR = "sampled_properties/"
# only used if property sampling grammar weights are "fitted"
FRONTIERS_PKL = "enumerationResults/neuralRecognizer_2021-05-18 15:27:58.504808_t=600.pkl"

def updateSavedPropertiesWithNewCacheTable(properties, propertiesPath):

    oldProperties = dill.load(open(propertiesPath, "rb"))

    for oldP in oldProperties:
        matchingProperty = [p for p in properties if p.name == oldP.name][0]
        for spec, propertyValue in matchingProperty.cachedTaskEvaluations.items():
            if spec in oldP.cachedTaskEvaluations:
                assert oldP.cachedTaskEvaluations[spec] == propertyValue
            else:
                oldP.cachedTaskEvaluations[spec] = propertyValue

    dill.dump(oldProperties, open(propertiesPath, "wb"))

    print("Updating cache table and rewriting properties at: {}".format(propertiesPath))
    return

def addPropSpecificPrimitives(args, propertyGrammar):

    propertyPrimitives = propertyGrammar.primitives
    if args["libraryName"] != "property_prims":
        toutputToList = Primitive("toutput_to_tlist", arrow(toutput, tlist(tint)), lambda x: x)
        tinputToList = Primitive("tinput_to_tlist", arrow(tinput, tlist(tint)), lambda x: x)
        propertyPrimitives = propertyPrimitives + [tinputToList, toutputToList]

    if args["propAddZeroToNinePrims"]:
        for i in range(10):
            if str(i) not in [primitive.name for primitive in propertyPrimitives]:
                propertyPrimitives.append(Primitive(str(i), tint, i))
    else:
        zeroToNinePrimitives = set([str(i) for i in range(10)])
        propertyPrimitives = [p for p in propertyPrimitives if p.name not in zeroToNinePrimitives]

    if args["propUseConjunction"]:
        propertyPrimitives.append(Primitive("and", arrow(tbool, tbool, tbool), lambda a: lambda b: a and b))

    expression2likelihood = propertyGrammar.expression2likelihood
    productions = [(expression2likelihood.get(p, 0.0), p) for p in propertyPrimitives]
    propertyGrammar = Grammar.fromProductions(productions)
    return propertyGrammar

def getPropertySamplingGrammar(baseGrammar, grammarName, args, pseudoCounts=1, seed=0):
    grammar = addPropSpecificPrimitives(args, baseGrammar)
    grammar = setPropertyGrammarWeights(grammar, grammarName, pseudoCounts=pseudoCounts, seed=seed)
    return grammar

def getPropertyGrammar(args):
    nameToPrimitives = {
        "josh_1": josh_primitives("1"),
        "josh_2": josh_primitives("2"),
        "josh_3": josh_primitives("3")[0],
        "josh_3.1": josh_primitives("3.1")[0],
        "josh_final": josh_primitives("final"),
        "property_prims": handWrittenProperties(),
        "dc_list_domain": bootstrapTarget_extra(),
        "josh_rich_0_10": josh_primitives("rich_0_10"),
    }
    propertyPrimitives = nameToPrimitives[args["libraryName"]]
    if args["libraryName"] != "property_prims":
        toutputToList = Primitive("toutput_to_tlist", arrow(toutput, tlist(tint)), lambda x: x)
        tinputToList = Primitive("tinput_to_tlist", arrow(tinput, tlist(tint)), lambda x: x)
        propertyPrimitives = propertyPrimitives + [tinputToList, toutputToList]

    if args["propAddZeroToNinePrims"]:
        for i in range(10):
            if str(i) not in [primitive.name for primitive in propertyPrimitives]:
                propertyPrimitives.append(Primitive(str(i), tint, i))
    else:
        zeroToNinePrimitives = set([str(i) for i in range(10)])
        propertyPrimitives = [p for p in propertyPrimitives if p.name not in zeroToNinePrimitives]

    if args["propUseConjunction"]:
        propertyPrimitives.append(Primitive("and", arrow(tbool, tbool, tbool), lambda a: lambda b: a and b))

    tasks = {
        "Lucas-old": lambda: retrieveJSONTasks("data/list_tasks.json") + sortBootstrap(),
        "josh_1": lambda: joshTasks("1"),
        "josh_2": lambda: joshTasks("2"),
        "josh_3": lambda: joshTasks("3"),
        "josh_3.1": lambda: joshTasks("3.1"),
        "josh_final": lambda: joshTasks("final"),
    }[args["dataset"]]()
    

    if "josh" in args["dataset"]:
        tasks = [t for t in tasks if int(t.name[:3]) < 81 and "_1" in t.name]
    tasks = [t for t in tasks if (t.request == arrow(tlist(tint), tlist(tint)) and isinstance(t.examples[0][1],list) and isinstance(t.examples[0][0][0],list))]
    print("{} tasks".format(len(tasks)))
    print("{} tasks".format(len(tasks)))

    expression2likelihood = {}
    productions = [(expression2likelihood.get(p, 0.0), p) for p in propertyPrimitives]
    propertyGrammar = Grammar.fromProductions(productions)
    propertyGrammar = setPropertyGrammarWeights(propertyGrammar, args["propSamplingGrammarWeights"])
    return propertyGrammar, tasks