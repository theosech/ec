import dill
import pandas as pd
import numpy as np

from dreamcoder.fragmentGrammar import FragmentGrammar
from dreamcoder.utilities import vprint, numberOfCPUs, parallelMap

from dreamcoder.properties.utils import createFrontiersWithInputsFromTask

DATA_DIR = "data/prop_sig/"
SAMPLED_PROPERTIES_DIR = "sampled_properties/"
MAX_NUM_SIM_TASKS_TO_PRINT = 5
THRESHOLD_POSTERIOR_SUM = 500
NEURAL_RECOGNITION_MODEL_PATH = "data/prop_sig/recognitionModels/josh_rich_enumerated_1/learned_9740_enumeratedFrontiers_ep=True_RS=None_RT=7200.pkl"

class ZeroPropertiesFound(Exception):
    pass

def getPropertySimTasksMatrix(helmholtzTasks, properties, taskPropertyValueToInt):
    """

    Returns:
        A matrix of size (len(allFrontiers), len(properties))
    """
    matrix = []
    for i,task in enumerate(helmholtzTasks):
        taskSig = [taskPropertyValueToInt[prop.getValue(task)] for prop in properties]
        matrix.append(taskSig)
    return np.array(matrix)


def getTaskSimilarFrontier(
    allFrontiers, 
    properties, 
    propertySimTasksMatrix, 
    valuesToInt, 
    allTasks,
    taskIdx, 
    grammar,
    filterSimilarProperties, 
    maxFractionSame, 
    nSim=20, 
    propertyToPriorDistribution=None, 
    onlyUseTrueProperties=True, 
    recomputeTasksWithTaskSpecificInputs=False,
    computePriorFromTasks=False,
    weightByPropertyPrior=True,
    verbose=False):
    """
    Returns:
        frontiersToUse (list): A list of Frontier objects of the nSim most similar tasks to task. List of frontier is used to fit unigram / fit 
        bigram / train recognition model
    """
    task = allTasks[taskIdx]
    vprint("\n------------------------------------------------ Task {} ----------------------------------------------------".format(task), verbose)
    vprint(task.describe(), verbose)
    try:
        simDf, matchingFrontiers, _ = createSimilarTasksDf(allTasks, taskIdx, allFrontiers, properties, propertySimTasksMatrix, propertyToPriorDistribution, valuesToInt, 
            onlyUseTrueProperties=onlyUseTrueProperties, filterSimilarProperties=filterSimilarProperties, maxFractionSame=maxFractionSame, 
            recomputeTasksWithTaskSpecificInputs=recomputeTasksWithTaskSpecificInputs, computePriorFromTasks=computePriorFromTasks, 
            weightByPropertyPrior=weightByPropertyPrior, verbose=verbose)
    except ZeroPropertiesFound:
        vprint("No properties found for task {}".format(task), verbose)
        return allFrontiers, [1 for i in range(len(allFrontiers))], False


    # check if any of the similar tasks programs are actually solutions for the task we want to solve
    solved = False
    for idx in simDf.head(nSim).index.values:
        simProgram = matchingFrontiers[idx].entries[0].program
        solved = task.check(simProgram, timeout=1)
        if solved:
            vprint("\nFound program solution for task {}: {}".format(task, simProgram), verbose)
            break


    vprint("\n{} Most Similar Prop Sig Tasks\n--------------------------------------------------------------------------------".format(nSim), verbose)
    frontiersToUse, frontierWeights = [], []
    
    toIter = enumerate(simDf.index.values) if nSim == (-1) else enumerate(simDf.head(nSim).index.values)
    totalPosteriorSum = 0
    for i,idx in toIter:
        if i < MAX_NUM_SIM_TASKS_TO_PRINT:
            vprint("\nTask similarity score: {}".format(simDf.loc[idx, "score"]), verbose)
            vprint(matchingFrontiers[idx].task.describe(), verbose)
            vprint("\nProgram ({}): {}".format(simDf.loc[idx, "programPrior"], simDf.loc[idx, "program"]), verbose)
        
        # evalReducedProgram = _reduceByEvaluating(simDf.loc[idx, "program"], {p.name: p for p in grammar.primitives})
        # vprint("\nEvaluation Reduced Program ({}): {}".format(grammar.logLikelihood(propertyFeatureExtractor.tasks[idx].request, evalReducedProgram), evalReducedProgram), verbose)

        frontiersToUse.append(matchingFrontiers[idx])
        frontierWeights.append(simDf.loc[idx, "score"])
       
    return frontiersToUse, frontierWeights, solved


def _getPriorBasedSimilarityScore(taskSig, trainTaskSig, propertyToPriorDistribution):

    taskProbsPerPropValue = propertyToPriorDistribution[taskSig, np.arange(propertyToPriorDistribution.shape[1])]
    trainTaskProbsPerPropValue = propertyToPriorDistribution[trainTaskSig, np.arange(propertyToPriorDistribution.shape[1])]   

    # The weight of each property p_i is 1 / the probability of (p_i(taskSig), p_i(trainTaskSig))
    weights = 1.0 / np.multiply(taskProbsPerPropValue,trainTaskProbsPerPropValue)
    
    # vector including only properties for which the 2 tasks have the same value
    samePropertyVals = np.equal(trainTaskSig,taskSig)
    similarityVector = np.ones(taskSig.shape[0])
    similarityVector = np.multiply(similarityVector[samePropertyVals], weights[samePropertyVals])

    # return np.sum(similarityVector)
    return np.sum(np.log(similarityVector))

def _getSimilarityScore(taskSig, trainTaskSig, onlyUseTrueProperties):

    similarityVector = np.zeros(taskSig.shape[0])
    samePropertyVals = np.equal(trainTaskSig,taskSig)

    if onlyUseTrueProperties:
        similarityVector[np.where((samePropertyVals) & (taskSig == 1))] = 1
        denominator = (taskSig == 1).sum()
    else:
        similarityVector[np.where((samePropertyVals))] = 1
        denominator = taskSig.shape[0]

    return np.sum(similarityVector.astype(int)) / denominator


def _filterProperties(properties, propTasksMatrix, maxFractionSame, save=False, filename=None):

    def fractionSame(a, b):
        return np.sum((a == b).astype(int)) / a.shape[0]

    assert len(properties) == propTasksMatrix.shape[1]

    filteredPropertyIds = []
    for i,p in enumerate(properties):
        taskSigToConsider = propTasksMatrix[:, i]
        if all([fractionSame(taskSigToConsider, propTasksMatrix[:, j]) < maxFractionSame for j in filteredPropertyIds]):
            filteredPropertyIds.append(i)

    filteredProperties = [properties[j] for j in filteredPropertyIds]
    print("Kept {} from {} properties".format(len(filteredProperties), propTasksMatrix.shape[1]))
    if save:
        path = DATA_DIR + SAMPLED_PROPERTIES_DIR + filename
        dill.dump(filteredProperties, open(path, "wb"))
        print("Saved filtered properties at: {}".format(path))
    return filteredProperties


def createSimilarTasksDf(
    allTasks, 
    taskIdx, 
    allFrontiers, 
    properties, 
    propertySimTasksMatrix, 
    propertyToPriorDistribution, 
    valuesToInt, 
    onlyUseTrueProperties, 
    filterSimilarProperties, 
    maxFractionSame, 
    recomputeTasksWithTaskSpecificInputs, 
    computePriorFromTasks,
    weightByPropertyPrior,
    verbose=False):
    """
    Args:
        task (Task): the task we want to solve.
        allFrontiers (list(Frontiers)): list of all the sampled frontiers
        properties (list(Property)): the list of properties to use for PropSim score
        propertySimTasksMatrix (np.ndarray): 2d numpy array of size (# of sampled frontiers, # of properties)
        propertyToPriorDistribution (np.ndarray): 2d numpy array of size (# unique property values, # properties)
        onlyUseTrueProperties (boolean): whether to calculate similarity score only using true properties of task or all of them. if 
        onlyUseTrueProperties any task with allTrue for all allTrue properties of task, will be deemed 100% similar even if for other
        properties their values differ.

    Returns:
        A dataframe where rows are tasks (indexed by task idx corresponding to propertyFeatureExtractor.tasks) with one column for the 
        similarity score of every task

    """
    task = allTasks[taskIdx]
    taskSig = np.array([valuesToInt[prop.getValue(task)] for prop in properties])
    if onlyUseTrueProperties:
        propertiesMask = (taskSig == valuesToInt["allTrue"])
    else:
        # only keep properties that aren't mixed for task we want to solve
        propertiesMask = np.logical_or(taskSig == valuesToInt["allTrue"], taskSig == valuesToInt["allFalse"])

    # filter properties only keeping ones with values desired for task to solve
    propertyToIdx = {p:i for i,p in enumerate(properties)}
    properties = [p for i,p in enumerate(properties) if propertiesMask[i]]

    if len(properties) == 0:
        raise ZeroPropertiesFound
    
    # this will only be true on the first iteration of iterative propSim where we use the same frontier for alll tasks
    if propertySimTasksMatrix is not None and propertyToPriorDistribution is not None:
        pass
    else:
        # we use the sampled programs but execute on inputs of tasks we want to solve
        if recomputeTasksWithTaskSpecificInputs: 
            allFrontiers = createFrontiersWithInputsFromTask(allFrontiers, task)

        propertySimTasksMatrix, propertyToPriorDistribution = _getSimTaskMatrixAndPropertyPriors(allTasks, allFrontiers, properties, valuesToInt, computePriorFromTasks)

        # update propertyToIdx to point to the idx of property in the new data structures
        propertyToIdx = {p:i for i,p in enumerate(properties)}
        vprint("new shape of propertyToPriorDistribution: {}".format(propertyToPriorDistribution.shape), verbose)
        vprint("new shape of propertySimTasksMatrix: {}".format(propertySimTasksMatrix.shape), verbose)

    # sorted properties by (1 / prior probability) of observed value
    propertyScores = list(propertyToPriorDistribution[taskSig[propertiesMask], [propertyToIdx[prop] for prop in properties]])
    sortedPropAndScores = sorted([(prop, score) for prop,score in zip(properties, propertyScores)], key=lambda x: x[1], reverse=False)
    properties = [el[0] for el in sortedPropAndScores]

    if filterSimilarProperties:
        # filter properties only keeping high-scoring "independent" ones
        idxs = [propertyToIdx[prop] for prop in properties]
        reorderedPropertySimTasksMatrix = propertySimTasksMatrix[:, idxs].copy()
        reorderedPropertySimTasksMatrix[reorderedPropertySimTasksMatrix != valuesToInt["allTrue"]] = 0
        properties = _filterProperties(properties=properties, propTasksMatrix=reorderedPropertySimTasksMatrix, maxFractionSame=maxFractionSame, save=False, filename=None)
        sortedPropAndScores = [(p,score) for p,score in sortedPropAndScores if p in properties]

    taskSig = np.array([valuesToInt[prop.getValue(task)] for prop in properties])

    n = min(20, len(properties))
    vprint("{} Highest scoring properties:".format(n), verbose)
    for prop,score in sortedPropAndScores[:n]:
        vprint("{} -> {} ({})".format(score, prop, prop.getValue(task)), verbose)
        # vprint(propertyToPriorDistribution[:, propertyToIdx[prop]], verbose)
        # print(propertySimTasksMatrix[:, propertyToIdx[prop]])

    data = propertySimTasksMatrix[:, [propertyToIdx[p] for p in properties]]
    df = pd.DataFrame(data=data)
    if weightByPropertyPrior:
        simSeries = df.apply(lambda row: _getPriorBasedSimilarityScore(taskSig, row, propertyToPriorDistribution[:, [propertyToIdx[p] for p in properties]]), axis=1)
    else:
        simSeries = df.apply(lambda row: _getSimilarityScore(taskSig, row, onlyUseTrueProperties), axis=1)
    simDf = simSeries.to_frame(name="score")

    # normalize score to make in range 0 to 1
    # simDf["score"] = simDf["score"] / simDf["score"].max()
    simDf["program"] = [allFrontiers[int(idx)].entries[0].program for idx in simDf.index]
    simDf["programPrior"] = [allFrontiers[int(idx)].entries[0].logPrior for idx in simDf.index]
    simDf["maxOutputListLength"] = [-max([len(o) for i,o in allFrontiers[int(idx)].task.examples]) for idx in simDf.index]
    simDf = simDf.drop_duplicates(subset=["program"])
    simDf = simDf.sort_values(["score", "programPrior", "maxOutputListLength"], ascending=False)

    # assert that the simDf row index is aligned with the index of the corresponding frontier
    for i,frontier in enumerate(allFrontiers):
        assert simDf.loc[i, "program"] == frontier.entries[0].program

    return simDf, allFrontiers, sortedPropAndScores


def getPriorDistributionsOfProperties(propertySimTasksMatrix, valuesToInt):
    """
    Calculates the prior distribution of each property from propertySimTasksMatrix.

    Args:
        propertySimTasksMatrix (np.ndarray): 2d numpy array of size (len(allFrontiers), len(properties))

    Returns:
        probs: (len(set(values_to_int.keys())), len(properties))
    """

    def getDistributionOfVector(vector, uniqueValues, pseudoCounts=1):

        # each element in counts corresponds to the frequency count of its idx
        counts = np.bincount(vector, minlength=len(uniqueValues))
        counts = counts + pseudoCounts
        normalizedCounts = counts / np.sum(counts)
        return normalizedCounts

    # matrix of size (number of unique values, properties) with probalities of property values
    uniqueValues = sorted(list(set(value for value in valuesToInt.values())))
    probs = np.zeros((len(uniqueValues), propertySimTasksMatrix.shape[1]))
    for i in range(propertySimTasksMatrix.shape[1]):
        probs[:, i] = getDistributionOfVector(propertySimTasksMatrix[:, i], uniqueValues)

    return probs


def _getSimTaskMatrixAndPropertyPriors(allTasks, frontiers, properties, valuesToInt, computePriorFromTasks):

    # print("Creating Similar Task Matrix")
    propertySimTasksMatrix = getPropertySimTasksMatrix([f.task for f in frontiers], properties, valuesToInt)
    # print("Finished Creating Similar Task Matrix with size: {}".format(propertySimTasksMatrix.shape))
    if computePriorFromTasks:
        propertyValsMatrix = getPropertySimTasksMatrix(allTasks, properties, valuesToInt)
        propertyToPriorDistribution = getPriorDistributionsOfProperties(propertyValsMatrix, valuesToInt)
    else:
        propertyToPriorDistribution = getPriorDistributionsOfProperties(propertySimTasksMatrix, valuesToInt)
        # print("propertyToPriorDistribution", propertyToPriorDistribution.shape)
    return propertySimTasksMatrix, propertyToPriorDistribution

def getPropSimGrammars(
    baseGrammars, 
    tasksToSolve,
    allTasks,
    sampledFrontiers, 
    properties,
    onlyUseTrueProperties, 
    nSim, 
    pseudoCounts, 
    weightedSim, 
    compressSimilar, 
    weightByProgramPrior, 
    recomputeTasksWithTaskSpecificInputs, 
    computePriorFromTasks, 
    filterSimilarProperties, 
    maxFractionSame, 
    valuesToInt,
    propSimIteration,
    weightByPropertyPrior,
    verbose):

    """
    Returns:
        task2FittedGrammar (dict): every key is a task (Task) and its value is its corresponding grammar (Grammar) fitted on the most similar tasks
        tasksSolved (set): set of tasks solved running PropSim (i.e. for which one of the 10,000 enumerated programs satisfied all I/O examples)
        task2SimilarFrontiers (dict): every key is a task (Task) and its value is the corresponding frontiers (list(Frontier)), one frontier for each similar task
    """

    task2SimilarFrontiers, task2FittedGrammar, tasksSolved = {}, {}, set()

    if not isinstance(baseGrammars, dict):
        baseGrammars = {t: baseGrammars for t in tasksToSolve}
    task2Grammar = baseGrammars

    if not isinstance(sampledFrontiers, dict):
        sampledFrontiers = {t: sampledFrontiers for t in tasksToSolve}
    task2Frontiers = sampledFrontiers

    if not isinstance(properties, dict):
        task2Properties = {t: properties for t in tasksToSolve}

    propertySimTasksMatrix, propertyToPriorDistribution = None, None
    # we can get away with computing once for all tasks if the below conditions are True
    if not recomputeTasksWithTaskSpecificInputs and computePriorFromTasks and propSimIteration == 0:
        propertySimTasksMatrix, propertyToPriorDistribution = _getSimTaskMatrixAndPropertyPriors(allTasks, sampledFrontiers[tasksToSolve[0]], properties, valuesToInt, True)

    # for taskIdx,task in enumerate(allTasks):

    def getTaskFittedGrammar(taskTuple):
        taskIdx, task = taskTuple
        if task in tasksToSolve:
            similarFrontiers, weights, solved = getTaskSimilarFrontier(task2Frontiers[task], task2Properties[task], propertySimTasksMatrix, valuesToInt, allTasks, taskIdx, task2Grammar[task], 
                filterSimilarProperties=filterSimilarProperties, maxFractionSame=maxFractionSame, nSim=nSim, propertyToPriorDistribution=propertyToPriorDistribution, 
                onlyUseTrueProperties=onlyUseTrueProperties, recomputeTasksWithTaskSpecificInputs=recomputeTasksWithTaskSpecificInputs, computePriorFromTasks=computePriorFromTasks, weightByPropertyPrior=weightByPropertyPrior, verbose=verbose)

            task2SimilarFrontiers[task] = similarFrontiers
            
            vprint("{} similar frontiers".format(len(similarFrontiers)), verbose)
            if compressSimilar:
                if len([f for f in similarFrontiers if not f.empty]) == 0:
                    eprint("No compression frontiers; not inducing a grammar this iteration.")
                else:
                    print(similarFrontiers)
                    maxScore = weights[0]
                    for w,f in zip(weights, similarFrontiers):
                        assert len(f.entries) == 1
                        print("max score: {}, w: {}".format(maxScore, w))
                        f.entries[0].logLikelihood = 0.0
                    print(similarFrontiers)
                    taskGrammar, fHelmFrontiers = FragmentGrammar.induceFromFrontiers(
                        task2Grammar[task],
                        similarFrontiers,
                        _=None,
                        topK=1,
                        topk_use_only_likelihood=False,
                        pseudoCounts=1.0,
                        aic=1.0,
                        structurePenalty=0.001,
                        a=0,
                        CPUs=1)
                    """
                    taskGrammar, compressionFrontiers = induceGrammar(task2Grammar[task], similarFrontiers,
                                                      topK=2,
                                                      pseudoCounts=1, a=3,
                                                      aic=1.0, structurePenalty=1.5,
                                                      topk_use_only_likelihood=False,
                                                      backend='ocaml', CPUs=numberOfCPUs(), iteration=0)
                    vprint("\nCompression frontiers for task {}: {}".format(compressionFrontiers, task), verbose)
                    """
            else:
                weights = weights if weightedSim else None
                vprint(similarFrontiers[0].task.describe(), verbose)
                taskGrammar = task2Grammar[task].insideOutside(similarFrontiers, pseudoCounts, iterations=1, frontierWeights=weights, weightByProgramPrior=weightByProgramPrior)
            
            vprint("\nGrammar after fitting for task {}:\n{}".format(task, taskGrammar), verbose)
            # task2FittedGrammar[task] = taskGrammar
            # if solved:
            #     tasksSolved.add(task)
            return taskGrammar, solved, similarFrontiers
        else:
            return None, None, None

    results = parallelMap(
        numberOfCPUs(),
        getTaskFittedGrammar,
        list(enumerate(allTasks)), memorySensitive=False)

    for t,result in zip(allTasks, results):
        grammar, solved, similarFrontiers = result
        if grammar is not None:
            task2FittedGrammar[t] = grammar
            if solved:
                tasksSolved.add(t)
            task2SimilarFrontiers[t] = similarFrontiers            

    return task2FittedGrammar, tasksSolved, task2SimilarFrontiers


