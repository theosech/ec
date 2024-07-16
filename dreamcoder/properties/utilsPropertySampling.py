import dill
import random

from dreamcoder.properties.property import Property

from dreamcoder.enumeration import multicoreEnumeration
from dreamcoder.likelihoodModel import UniqueTaskSignatureScore, TaskSurprisalScore, GeneralUniqueTaskSignatureScore

OUTPUT_STR = "$0"
MIN_LOG_PRIOR = -11
FRONTIERS_PKL = "enumerationResults/neuralRecognizer_2021-05-18 15:27:58.504808_t=600.pkl"

def setPropertyGrammarWeights(baseGrammar, grammarName, pseudoCounts=1, seed=0):
    if grammarName == "random":
        random.seed(seed)
        grammar = baseGrammar.randomWeights(r=lambda oldWeight: -1 * random.uniform(0,5))
    elif grammarName == "fitted":
        frontiers, times = dill.load(open(FRONTIERS_PKL, "rb"))
        frontiersToFitOn = [f for f in frontiers if (len(f.entries) > 0 and baseGrammar.logLikelihood(f.task.request, f.topK(1).entries[0].program) > MIN_LOG_PRIOR)]
        print("Fitting on {} frontiers with logPrior > {}".format(len(frontiersToFitOn), MIN_LOG_PRIOR))
        grammar = baseGrammar.insideOutside(frontiersToFitOn, pseudoCounts=pseudoCounts)
    elif grammarName == "same":
        grammar = baseGrammar
    else:
        raise Exception("Provided sampling grammar weights argument: {} is invalid".format(grammarName))
    return grammar

def prop_sampling_options(parser):
    parser.add_argument("--propUseConjunction", action="store_true", default=False)
    parser.add_argument("--propAddZeroToNinePrims", action="store_true", default=False)
    parser.add_argument("--propEnumerationTimeout",default=1,type=float)
    parser.add_argument("--propSamplingMethod", default="unique_task_signature", choices=[
        "per_task_discrimination",
        "unique_task_signature"
        ])
    parser.add_argument("--propSamplingGrammarWeights", default="same", choices=[
        "same",
        "fitted",
        "random"
        ])
    return parser

def enumerateProperties(propertyGrammar, tasksToSolve, propertyRequest, propScoringMethod, propSolver, propCPUs, propEnumerationTimeout, allTasks=None):

    # if we sample properties by "unique_task_signature" we don't need to enumerate the same properties
    # for every task, as whether we choose to include the property or not depends on all tasks.
    if propScoringMethod == "unique_task_signature":
        likelihoodModel = UniqueTaskSignatureScore(timeout=0.1, tasks=allTasks)
        tasksToSolve = tasksToSolve[0:1]
    elif propScoringMethod == "general_unique_task_signature":
        likelihoodModel = GeneralUniqueTaskSignatureScore(timeout=0.1, tasks=allTasks)
    elif propScoringMethod == "per_task_surprisal":
        likelihoodModel = TaskSurprisalScore(timeout=0.1, tasks=allTasks)
    else:
        raise NotImplementedError

    print("Enumerating with {} CPUs".format(propCPUs))
    if isinstance(propertyGrammar, dict):
        for t in tasksToSolve:
            print("Task: {} ({})", t.name, t.request)
            print("Task Grammar: {}".format(propertyGrammar[t]))
    else:
        print("Grammar: {}".format(propertyGrammar))

    frontiers, times, pcs, likelihoodModel = multicoreEnumeration(propertyGrammar, tasksToSolve, solver=propSolver,maximumFrontier= int(10e7),
                                                 enumerationTimeout= propEnumerationTimeout, CPUs=propCPUs,
                                                 evaluationTimeout=0.1,
                                                 testing=True, likelihoodModel=likelihoodModel)
    
    # frontiers, times, pcs = enumerateForTasks(propertyGrammar, tasksToSolve,
    #                                             timeout=args["propEnumerationTimeout"], CPUs=args["propCPUs"],maximumFrontiers= {t:int(10e7) for t in tasksToSolve},
    #                                             evaluationTimeout=1, upperBound=2000,
    #                                             testing=True, likelihoodModel=likelihoodModel)
    
    # def parseAndScore(entry, propertyTask, likelihoodModel):
    #    program = Program.parse(entry["programs"][0])
    #    keep, propScore = likelihoodModel.score(program, propertyTask, str(program), propertyTask.request)
    #    if keep:
    #        prop = Property(program=program, name=str(program), request=propertyTask.request, score=propScore)
    #        return True, prop
    #    return False, None


    # response = enumerateFromOcamlGrammar(tasksToSolve, propertyGrammar, enumerationTimeout[args["propEnumerationTimeout"]])
    # properties = parallelMap(args["propCPUs"], lambda entry: parseAndScore(entry, request, likelihoodModel), response, memorySensitive=True)

    print("frontiers: ")
    for f in frontiers:
        print(f.task.request)
        print(f.entries)

    properties = []
    if propScoringMethod == "unique_task_signature":
        assert (len(frontiers) == 1)
        for entry in frontiers[0].entries:
            if OUTPUT_STR in str(entry.program):
                prop = Property(program=entry.program.evaluate([]), 
                            request=propertyRequest, 
                            name=str(entry.program), 
                            logPrior=entry.logPrior, 
                            score=entry.logLikelihood)
                properties.append(prop)

    else:
        for f in frontiers:
            for entry in f.entries:
                if OUTPUT_STR in str(entry.program):
                    prop = Property(program=entry.program.evaluate([]), 
                                request=propertyRequest, 
                                name=str(entry.program), 
                                logPrior=entry.logPrior, 
                                score=entry.logLikelihood)
                    properties.append(prop)

    return properties, likelihoodModel