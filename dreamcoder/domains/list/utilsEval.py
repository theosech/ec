import datetime
import dill
import matplotlib.pyplot as plt
import numpy as np
import pickle

# from dreamcoder.domains.list.utilsPlotting import plotFrontiers
from dreamcoder.enumeration import multicoreEnumeration
from dreamcoder.grammar import Grammar
from dreamcoder.utilities import vprint

def loadEnumerationResults(filename):
    try:
        frontiers, times, taskToPrograms = pickle.load(open("enumerationResults/{}".format(filename), "rb"))
        # toPlot = []
        # for t,numPrograms in taskToPrograms.items():
        #     if numPrograms is not None:
        #         print("{}: {}".format(t.name, numPrograms))
        #         toPlot.append(numPrograms)
        # # plt.hist(toPlot)
        # # plt.show()
        # print(filename, np.median(toPlot), np.mean(toPlot))
    except ValueError:
        frontiers, times = pickle.load(open("enumerationResults/{}".format(filename), "rb"))
    return frontiers, times

def getRecognizerTaskGrammars(trainedRecognizer, tasks):
    grammars = {task: trainedRecognizer.grammarOfTask(task)
                for task in tasks}
    #untorch seperately to make sure you filter out None grammars
    grammars = {task: grammar.untorch() for task, grammar in grammars.items() if grammar is not None}
    return grammars


def enumerateFromGrammar(grammars, tasks, modelName, enumerationTimeout, solver, CPUs, maximumFrontier, leaveHoldout=False, save=False):
    bottomUpFrontiers, allRecognitionTimes, _, _ = multicoreEnumeration(grammars, tasks, _=None,
                             enumerationTimeout=enumerationTimeout,
                             solver=solver,
                             CPUs=CPUs,
                             maximumFrontier=maximumFrontier,
                             verbose=True,
                             evaluationTimeout=1.0,
                             testing=True,
                             likelihoodModel=None,
                             leaveHoldout=leaveHoldout)
    if save:
        savePath = "enumerationResults/{}_{}_t={}.pkl".format(modelName, datetime.datetime.now(), enumerationTimeout)
        with open(savePath, "wb") as handle:
            dill.dump((bottomUpFrontiers, allRecognitionTimes), handle)
        print("Saved enumeration results at: {}".format(savePath))
    return bottomUpFrontiers, allRecognitionTimes

def enumerateFromGrammars(args, tasks, allGrammars, modelNames, save, plotName):

    enumerationTimeout, solver, maximumFrontier, CPUs = args.pop("enumerationTimeout"), args.pop("solver"), args.pop("maximumFrontier"), args.pop("CPUs")
    
    enumerationResults = []
    for g, modelName in zip(allGrammars, modelNames):
         print("grammar for first task: {}".format(g if isinstance(g, Grammar) else list(g.values())[0]))
         bottomUpFrontiers, allRecognitionTimes = enumerateFromGrammar(g, tasks, modelName, enumerationTimeout, solver, CPUs, maximumFrontier, leaveHoldout=True, save=save)
         enumerationResults.append((bottomUpFrontiers, allRecognitionTimes))
         nonEmptyFrontiers = [f for f in bottomUpFrontiers if not f.empty]
         numTasksSolved = len([f.task for f in nonEmptyFrontiers if f.task.check(f.topK(1).entries[0].program, timeout=1.0, leaveHoldout=False)])
         print("Enumerating from {} grammars for {} seconds: {} / {} actually true for holdout example".format(modelName, enumerationTimeout, numTasksSolved, len(nonEmptyFrontiers)))
    
    # plotFrontiers(modelNames, enumerationResults=enumerationResults, save=True, plotName=plotName)
    return

def enumerationProxy(task2FittedGrammars, tasks, modelNames, verbose=False):
    """
    Prints the log posterior of tasks under different grammars

    Args:
        task2FittedGrammars (list): List of either task2grammar dictionaries or Grammar if grammar is the same for all tasks
        tasks (list(Task)): List of tasks for which to calculate log posterior
        modelNames (list(str)): List of model names corresponding to task2FittedGrammars

    Returns:
        modelToTaskToLogposterior (dict): dictionary with model nameas as keys and dictionaries of (task, logPosterior) entries as values
    """

    modelToLogPosteriors = {modelName:[] for modelName in modelNames}
    tasksWithGtPrograms = [t for t in tasks if t.program is not None]
    for task in tasksWithGtPrograms:
        vprint("\n-------------------------------------------------------------------------------", verbose)
        vprint(task.describe(), verbose)
        vprint("Ground Truth Program: {}".format(task.program), verbose)
        vprint("---------------------------------------------------------------------------------", verbose)
        for task2grammar, modelName in zip(task2FittedGrammars, modelNames):
            taskGrammar = task2grammar if isinstance(task2grammar, Grammar) else task2grammar[task]
            logPosterior = taskGrammar.logLikelihood(task.request, task.program)
            modelToLogPosteriors[modelName].append(logPosterior)
            vprint("{}: {}".format(modelName, logPosterior), verbose)

    for modelName, logPosteriors in modelToLogPosteriors.items():
        vprint("Mean {} Log posterior: {}".format(modelName, sum(logPosteriors) / len(logPosteriors)), verbose)

    return {modelName: {t: logPosterior for t,logPosterior in zip(tasksWithGtPrograms, logPosteriors)} for modelName,logPosteriors in modelToLogPosteriors.items()}

def cumulativeNumberOfTasksSolved(enumerationFilename, enumerationTime):
    frontiers, times = loadEnumerationResults(enumerationFilename)    
    satisfiesHoldout = lambda f: len(f.entries) > 0 and f.task.check(f.topK(1).entries[0].program, timeout=1.0, leaveHoldout=False)
    solvedTasks = set(f.task for f in frontiers if satisfiesHoldout(f))
    approximateMAPSortedTimes = sorted([time for task,time in times.items() if task in solvedTasks])

    counts = 0
    for time in approximateMAPSortedTimes:
        if time > enumerationTime:
            return counts
        counts += 1
    return counts
