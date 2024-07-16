from dreamcoder.dreaming import *
from dreamcoder.properties.propertySignatureExtractor import PropertySignatureExtractor
from dreamcoder.properties.propSim import getPropertySimTasksMatrix, getPriorDistributionsOfProperties

VALUES_TO_INT = {"allFalse":0, "allTrue":1, "mixed":2}

def main(propertyGrammar, tasks, propertyRequest, args):
    # instatiating the feature extractor samples properties if args["propToUse"] == "sample"
    featureExtractor = PropertySignatureExtractor(tasksToSolve=tasks, testingTasks=[], H=64, embedSize=16, helmholtzTimeout=0.001, helmholtzEvaluationTimeout=0.001,
            cuda=False, 
            propUseEmbeddings=args["propUseEmbeddings"],
            propToUse=args["propToUse"],
            propScoringMethod=args["propScoringMethod"],
            propSolver=args["propSolver"],
            propCPUs=args["propCPUs"],
            propEnumerationTimeout=args["propEnumerationTimeout"],
            propFilename=args["propFilename"],
            propAddZeroToNinePrims=args["propAddZeroToNinePrims"],
            propUseConjunction=args["propUseConjunction"],
            propertyGrammar=propertyGrammar, grammar=None, propertyRequest=propertyRequest)

    propertySimTasksMatrix = getPropertySimTasksMatrix(tasks, featureExtractor.properties, VALUES_TO_INT)
    propertyToPriorDistribution = getPriorDistributionsOfProperties(propertySimTasksMatrix, VALUES_TO_INT)
    for i,p in enumerate(featureExtractor.properties):
        p.setPropertyValuePriors(propertyToPriorDistribution[:, i], VALUES_TO_INT)

    for t in tasks:
        propertyScores = [(p, p.getPropertyValuePrior(p.getValue(t))) for p in featureExtractor.properties]
        sortedPropertyScores = sorted([(p,score) for p,score in propertyScores], key=lambda el: el[1])
        print("\n{}".format(t.describe()))
        for p,score in sortedPropertyScores[:10]:
            print(p.name, score, p.getValue(t))
