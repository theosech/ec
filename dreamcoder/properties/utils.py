import copy
import dill
import json
import os

from dreamcoder.dreaming import helmholtzEnumeration
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.program import Program
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tlist, tint, tbool
from dreamcoder.utilities import *

DATA_DIR = "data/prop_sig/"
MAX_OUTPUT_LENGTH = 1000

def convertToPropertyTasks(tasks, propertyRequest):
    propertyTasks = []
    for i,t in enumerate(tasks):
        # if i == 1981:
        #     continue
        tCopy = copy.deepcopy(t)
        tCopy.specialTask = ("property", None)
        tCopy.request = propertyRequest
        # tCopy.examples = [io for io in tCopy.examples]
        # tCopy.examples = [(tuplify([io[0][0], (io[1],)]), True) for io in tCopy.examples]
        propertyTasks.append(tCopy)
    return propertyTasks

def createFrontiersWithInputsFromTask(frontiers, task):
    
    newFrontiers = []
    frontiers = [f for f in frontiers if len(f.entries) > 0]

    for frontier in frontiers:
        func = frontier.entries[0].program.evaluate([])
        try:
            newExamples = []
            for i,_ in task.examples:
                o = task.predict(func, i)
                if len(o) > MAX_OUTPUT_LENGTH:
                    # print("Output too long")
                    # print("Program: {}".format(frontier.entries[0].program))
                    # print("input: {}".format(i))
                    raise ValueError()
                newExamples.append((i,o))
        except Exception as e:
            # print(e)
            continue
        newTask = Task(frontier.task.name, frontier.task.request, newExamples, features=None, cache=False)
        newFrontier = Frontier(frontier.entries, newTask)
        newFrontiers.append(newFrontier)

    print("Dropping {} out of {} due to error when executing on specified inputs".format(len(frontiers) - len(newFrontiers), len(frontiers)))
    return newFrontiers


# def _reduceByEvaluating(p, primitives):
#     evalReduced, p = p.evalReduce(primitives)
#     while evalReduced:
#         evalReduced, temp = p.evalReduce(primitives)
#         if temp == p:
#             break
#         else:
#             p = temp
#     return p


def _makeTaskFromProgram(program, request, featureExtractor, differentOutputs=True, filterIdentityTask=True):
    task = featureExtractor.taskOfProgram(program, request)
    if task is None:
        return None
    else:
        if differentOutputs:
            if all([o == task.examples[0][1] for i,o in task.examples]):
                return None
        if filterIdentityTask:
            if all([i[0] == o for i,o in task.examples]):
                return None
    return task

def _enumerateFromOcamlGrammar(tasks, grammar, enumerationTimeout, special):
    requests = list({t.request for t in tasks})
    request = requests[0]
    assert len(requests) == 1
    inputs = list({tuplify(xs)
                       for t in tasks if t.request == request
                       for xs, y in t.examples})
    print("Enumerating helmholtz tasks for {} seconds".format(enumerationTimeout))
    response = helmholtzEnumeration(grammar, request, inputs, enumerationTimeout, _=None, special="unique", evaluationTimeout=0.004, maximumSize=99999999)
    return response

def enumerateHelmholtzOcaml(tasks, grammar, enumerationTimeout, CPUs, featureExtractor, save=False, libraryName=None, datasetName=None):
    response = _enumerateFromOcamlGrammar(tasks, grammar, enumerationTimeout, special="unique")
    print("Response length: {}".format(len(response)))
    frontiers = []
    print("First 200 characters of response: {}".format(response[:200]))
    response = json.loads(response.decode("utf-8"))
       
    def parseAndMakeTaskFromProgram(entry, request, featureExtractor):
        program = Program.parse(entry["programs"][0])
        task = _makeTaskFromProgram(program, request, featureExtractor, differentOutputs=True, filterIdentityTask=True)
        if task is None:
            return None
        frontier = Frontier([FrontierEntry(program=Program.parse(p), logPrior=entry["ll"], logLikelihood=0.0) for p in entry["programs"]], task=task)
        return frontier

    requests = list(set([t.request for t in tasks]))
    assert len(requests) == 1
    request = requests[0]
    frontiers = parallelMap(CPUs, lambda entry: parseAndMakeTaskFromProgram(entry, request, featureExtractor), response, memorySensitive=True)
    frontiers = [f for f in frontiers if f is not None] 
    print("{} Frontiers after filtering".format(len(frontiers)))
    
    if save:
        savePath = "{}/helmholtz_frontiers/{}_enumerated/{}_with_{}-inputs.pkl".format(DATA_DIR, libraryName, len(frontiers), datasetName)
        os.makedirs("{}/helmholtz_frontiers/{}_enumerated".format(DATA_DIR, libraryName), exist_ok=True)
        dill.dump(frontiers, open(savePath, "wb"))
        print("Saving frontiers at: {}".format(savePath))
        return frontiers, savePath
    return frontiers, None

def loadEnumeratedTasks(filename, primitives=None, numExamples=11):
    
    path = "data/prop_sig/helmholtz_frontiers/{}".format(filename)
    with open(path, "rb") as f:
        frontiers = dill.load(f)

        filteredFrontiers = []
        numTooLong, numWrongType = 0, 0
        for j,f in enumerate(frontiers):

            try:
                p = Program.parse(str(f.topK(1).entries[0].program), primitives={p.name:p for p in primitives})
            except:
                continue

            # assert every frontier is of the desired type
            assert f.task.request == arrow(tlist(tint), tlist(tint))
            # exclude examples where the output is too large
            wrongType = any([(not isinstance(o, list)) or (not isinstance(i[0], list)) for i,o in f.task.examples])
            if wrongType:
                numWrongType += 1
                continue

            examples = [(i,o) for i,o in f.task.examples if len(o) < 50]
            # if sampled task has more than 11 examples keep only the first 11
            if len(examples) < numExamples:
                numTooLong += 1
                continue

            f.task.examples = examples[:numExamples]
            filteredFrontiers.append(f)

    print("Removed {} tasks cause they had too long outputs".format(numTooLong))
    print("Removed {} tasks cause they were the wrong type".format(numWrongType))
    print("{} total frontiers".format(len(filteredFrontiers)))
    return filteredFrontiers

def property_options(parser):
    parser.add_argument("--equalWeightProperties", action="store_true", default=False)
    parser.add_argument("--compressSimilar", action="store_true", default=False)
    parser.add_argument("--propSim", action="store_true", default=False)
    parser.add_argument("--helmEnumerationTimeout", type=int, default=1)
    parser.add_argument("--numHelmFrontiers", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--earlyStopping", action="store_true", default=False)
    parser.add_argument("--maxFractionSame", type=float, default=1.0)
    parser.add_argument("--filterSimilarProperties", action="store_true", default=False)
    parser.add_argument("--computePriorFromTasks", action="store_true", default=False)
    parser.add_argument("--nSim", type=int, default=50)
    parser.add_argument("--propPseudocounts", type=int, default=1)
    parser.add_argument("--onlyUseTrueProperties", action="store_true", default=False)
    parser.add_argument("--weightByPropertyPrior", action="store_true", default=False)
    parser.add_argument("--weightByProgramPrior", action="store_true", default=False)
    parser.add_argument("--weightedSim", action="store_true", default=False)
    parser.add_argument("--taskSpecificInputs", action="store_true", default=False)
    parser.add_argument("--propCPUs", type=int, default=numberOfCPUs())
    parser.add_argument("--propSolver",default="python", type=str)
    parser.add_argument("--propScoringMethod", default="unique_task_signature", choices=[
        "per_task_discrimination",
        "unique_task_signature",
        "general_unique_task_signature",
        "per_similar_task_discrimination",
        "per_task_surprisal"
        ])
    parser.add_argument("--propToUse", default="sample", choices=[
        "handwritten",
        "preloaded",
        "sample"
        ])
    parser.add_argument("--propFilename", type=str, default=None)
    parser.add_argument("--propUseEmbeddings", action="store_true", default=False)
    parser.add_argument("--valuesToInt", default={"allFalse":0, "allTrue":1, "mixed":2})

    # parser.add_argument("--propDreamTasks", action="store_true", default=False)
    # parser.add_argument("--debug", action="store_true", default=False)
    # parser.add_argument("--singleTask", action="store_true", default=False)
    # parser.add_argument("--save", action="store_true", default=False)
    # parser.add_argument("--propNumIters", type=int, default=1)
    # parser.add_argument("--hmfSeed", type=int, default=None)
    return parser