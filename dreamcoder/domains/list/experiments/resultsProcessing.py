import dill

from dreamcoder.domains.list.listPrimitives import basePrimitives, primitives, McCarthyPrimitives, bootstrapTarget_extra, no_length, josh_primitives
from dreamcoder.domains.list.handwrittenProperties import handWrittenProperties, tinput, toutput
from dreamcoder.grammar import Context, Grammar
from dreamcoder.program import *
from dreamcoder.type import *
from dreamcoder.utilities import eprint

import math
# import matplotlib.pyplot as plt
import numpy as np
import torch

def resume_from_path(resume):
    try:
        resume = int(resume)
        path = checkpointPath(resume)
    except ValueError:
        path = resume
    with open(path, "rb") as handle:
        result = dill.load(handle)
    resume = len(result.grammars) - 1
    eprint("Loaded checkpoint from", path)
    grammar = result.grammars[-1] if result.grammars else grammar
    return result, grammar, result.grammars[0]

def getLearnedProductions(result):
    currentProductions = set()
    learnedProductions = {}
    for i,grammar in enumerate(result.grammars):
        learnedProductions[i] = {}
        productions = set([str(p[2]) for p in grammar.productions])
        if len(list(currentProductions)) == 0:
            currentProductions = productions
        else:
            # print('----------------------------------------------------{}-----------------------------------------------------'.format(i))        
            newProductions = productions.difference(currentProductions)
            for j,production in enumerate(newProductions):
                learnedProductions[i][j] = production
                # print(j, production)
        currentProductions = currentProductions.union(productions)
    return learnedProductions


def getTrainFrontier(resumePath, n):
    result, resumeGrammar, firstGrammar = resume_from_path(resumePath)
    firstFrontier = [frontiers[0] for (key,frontiers) in result.frontiersOverTime.items() if len(frontiers[0].entries) > 0]
    allFrontiers = [frontier for (task,frontier) in result.allFrontiers.items() if len(frontier.entries) > 0]
    # expandedFrontier = expandFrontier(firstFrontier, n)
    print(result.learningCurve)

    testingTasks = result.getTestingTasks()
    trainTasks = result.taskSolutions.keys()

    learnedProductions = getLearnedProductions(result)
    return firstFrontier, allFrontiers, result.frontiersOverTime, resumeGrammar, firstGrammar, result.recognitionModel, learnedProductions, testingTasks, trainTasks


def scoreProgram(p, recognizer=None, grammar=None, task=None):
    if p is None:
        return -100

    if recognizer is not None:
        grammar = recognizer.grammarOfTask(task)
        if hasattr(recognizer, "lrModel"):
            if recognizer.lrModel:
                pass
            else:
                grammar = grammar.untorch()
        else:
            grammar = grammar.untorch()
    ll = grammar.logLikelihood(task.request, p)
    return ll

def evaluateGrammars(frontiersOverTime, tasks, grammar1=None, grammar2=None, recognizer1=None, recognizer2=None):

    lastFrontier = {task:allFrontiers[-1] for task,allFrontiers in frontiersOverTime.items() if (len(allFrontiers[-1]) > 0) and task in tasks}

    print("{} Tasks have solutions for last frontier".format(len(lastFrontier.keys())))

    averageLLBefore, averageLLAfter = 0,0
    for task, frontier in lastFrontier.items():

        llBefore = scoreProgram(frontier.entries[0].program, task=task, grammar=grammar1, recognizer=recognizer1)
        llAfter = scoreProgram(frontier.entries[0].program, task=task, grammar=grammar2, recognizer=recognizer2)
        if llAfter == 0:
            pass
            # print("{}: {}".format(frontier.task, llBefore))
        else:
            pass
            # print("{}: {} -> {}".format(frontier.task.name, llBefore, llAfter))
        # print("Program: {}\n".format(frontier.entries[0].program))
        averageLLBefore += llBefore
        averageLLAfter += llAfter

    if llAfter == 0:
        print('Average LL: {}'.format(averageLLBefore / len(lastFrontier)))
    else:
        print('Average LL: {} -> {}'.format(averageLLBefore / len(lastFrontier), (averageLLAfter / len(lastFrontier))))

    return


def evaluateRecognizers(grammar, ecResults, recognizerNames=None, iteration=-1):

    if recognizerNames is None:
        recognizerNames = [str(i) for i in range(len(ecResults))]

    tasks = list(ecResults[0].frontiersOverTime.keys())
    
    meanLogPosteriors = [0.0 for i in range(len(ecResults))]
    numTasksSolved = 0.0

    for task in tasks:

        programs = []
        for i,ecResult in enumerate(ecResults):

            # # backwards compatibility hack for RecognitionModels saved before code for LR was added
            # if not hasattr(ecResult.recognitionModel, "lrModel"):
            #     ecResult.recognitionModel.lrModel = False

            bestFrontier = ecResult.frontiersOverTime[task][iteration].topK(1)
            if len(bestFrontier) > 0:
                program = bestFrontier.entries[0].program
            else:
                program = None
            programs.append(program)
        
        print("----------------------------------------------------------------------------------------------------------------------------------")
        print("\n\nTask {}".format(task))
        for i in range(min(len(task.examples), 10)):
            print("{} -> {}".format(task.examples[i][0][0], task.examples[i][1]))
        print("----------------------------------------------------------------------------------------------------------------------------------")

        if all([p is None for p in programs]):
            continue
        else:

            numTasksSolved += 1
            bestProgram = max([(program,scoreProgram(program, ecResult.recognitionModel, grammar=grammar, task=task)) for program in programs],
                key=lambda x: x[1])[0]

            bestPriorProgram, logPrior = max([(program,scoreProgram(program, None, grammar=grammar, task=task)) for program in programs],
                key=lambda x: x[1])

            print("Best Program LogPrior: {} -> {}".format(bestPriorProgram, logPrior))
            print("----------------------------------------------------------------------------------------------------------------------------------")
            previousLogPosterior = 0.0

            for i,p in enumerate(programs):
                logPosterior = scoreProgram(bestProgram if p is None else p, ecResults[i].recognitionModel, grammar=grammar, task=task)

                # if logPosterior > previousLogPosterior:
                #     print("{}: {}, {}: {}".format(recognizerNames[1], logPosterior, recognizerNames[0], previousLogPosterior))
                meanLogPosteriors[i] += logPosterior
                logPosteriorStr = str(logPosterior) if p is not None else "bestProgramLogPosterior: {}".format(logPosterior)
                previousLogPosterior = logPosterior
                print("{}: {} ({})".format(recognizerNames[i], p, logPosteriorStr))




    for i, name in enumerate(recognizerNames):
        print("{} mean LogPosterior: {}".format(recognizerNames[i], meanLogPosteriors[i] / numTasksSolved))

    return

def evaluateRecognizersForTask(baseGrammar, ecResults, recognizerNames=None, taskToInvestigate=None, request=None, productionToInvestigate=None, printGrammar=False):

    print("\nGrammars\n--------------------------------------------------------------------------------------------------------------------------")

    if recognizerNames is None:
        recognizerNames = [str(i) for i in range(len(ecResults))]

    tasks = list(ecResults[0].frontiersOverTime.keys())
    task = [t for t in tasks if t.name == taskToInvestigate][0]

    grammars = []
    for i, ecResult in enumerate(ecResults):
        grammar = ecResult.recognitionModel.grammarOfTask(task).untorch()
        grammars.append(grammar)

        if request is not None:
            table = grammar.buildCandidates(request, Context.EMPTY, [], normalize=True, returnTable=True, returnProbabilities=True, mustBeLeaf=False)
            table = {key.name:value for key,value in table.items()}
            print("\nRequest: {}, Recognizer: {}".format(request, recognizerNames[i]))
            if productionToInvestigate is not None:
                print("p({}): {}".format(productionToInvestigate, table[productionToInvestigate][0]))
            else:
                print(table)

    priorTable = baseGrammar.buildCandidates(request, Context.EMPTY, [], normalize=True, returnTable=True, returnProbabilities=True, mustBeLeaf=False)
    priorTable = {key.name:value for key,value in priorTable.items()}
    if productionToInvestigate is not None:
        print("\nPrior prob for Request: {}".format(request))
        print("p({}): {}".format(productionToInvestigate, priorTable[productionToInvestigate][0]))

    if printGrammar:
        print("\n")
        for i,el in enumerate(baseGrammar.productions):
            _,_,production = el

            toDisplay = "{}: {:.2f} (logPrior) ".format(production, float(baseGrammar.productions[i][0]))
            for j,g in enumerate(grammars):
                toDisplay += "| {:.2f} ({}) ".format(float(g.productions[i][0]), recognizerNames[j])
            print(toDisplay)


        uniformGrammarPrior = grammar.logLikelihood(task.request, program)
        print("Uniform Grammar Prior: {}".format(uniformGrammarPrior))
        logVariableGrammar = Grammar(2.0, [(0.0, p.infer(), p) for p in grammar.primitives], continuationType=None)
        logVariableGrammarPrior = logVariableGrammar.logLikelihood(task.request, program)

def viewResults(rec):

    pickleFile = "experimentOutputs/jrule/2021-04-29T00:06:25.721563/jrule_arity=3_BO=False_CO=False_dp=False_doshaping=False_ES=1_ET=600_epochs=99999_HR=1.0_it=1_MF=10_parallelTest=False_RT=7200_RR=False_RW=False_st=False_STM=True_TRR=default_K=2_topkNotMAP=False_tset=S12_DSL=False.pickle"
    result1, resumeGrammar, _ = resume_from_path(pickleFile)

    # pickleFile = "experimentOutputs/jrule/2021-04-27T20:49:31.738479/jrule_arity=3_BO=False_CO=False_dp=False_doshaping=False_ES=1_ET=600_epochs=99999_HR=1.0_it=1_MF=10_parallelTest=False_RT=7200_RR=False_RW=False_st=False_STM=True_TRR=default_K=2_topkNotMAP=False_tset=S12_DSL=False.pickle"
    # result2, resumeGrammar, _ = resume_from_path(pickleFile)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result2, resumeGrammar, _ = resume_from_path(pickleFile)
    result2.recognitionModel = rec

    recognizerNames = ['rnn_encoded', 'rnn_encoded_2']
    evaluateRecognizers(resumeGrammar, [result1, result2], recognizerNames)

    # requests = [tlist(tint), tint, tint, tint, tint, tlist(tint)]
    # productions = ["map", "2", "3", "4", "index", "range"]
    
    # for i,prod in enumerate(productions):
    #     evaluateRecognizersForTask(resumeGrammar, 
    #                                [baselineResult_2, baselinePlusPropSigResult_2], 
    #                                recognizerNames, 
    #                                taskToInvestigate="005_1", 
    #                                request=requests[i], 
    #                                productionToInvestigate=prod, 
    #                                printGrammar=False)




    # print(extractor.embedding(torch.LongTensor([0,1,2,0])))

    # print(propSigOnlyResult.recognitionModel.generativeModel)

    # request = arrow(tlist(tint), tlist(tint))

    # numSamples = 100

    # samples = propSigOnlyResult.recognitionModel.sampleManyHelmholtz(requests=[request], N=numSamples, CPUs=1)
    # for frontier in samples:
    #     print("Task:{}\nProgram: {}\n".format("\n".join(["{} -> {}".format(i[0],o) for i,o in frontier.task.examples]), frontier.entries[0].program))


    # prims = bootstrapTarget_extra(useInts=True)
    # grammar = Grammar.uniform(prims)
    # print(grammar)
    # request = arrow(tlist(tint), tlist(tint))
    # print("Sampling with request: {}".format(request))

    # for i in range(numSamples):
    #     sample = grammar.sample(request=request)
    #     print("Program: {}".format(sample))

    # print("--------------------------------------------------------------------------------------------------")

    # toutputToList = Primitive("tlist_to_toutput", arrow(tlist(tint), toutput), lambda x: x)
    # tinputToList = Primitive("tinput_to_tlist", arrow(tinput, tlist(tint)), lambda x: x)
    # prims = prims + [toutputToList, tinputToList]
    # grammar = Grammar.fromProductions([(3.0, p) if p.name == "tinput_to_tlist" else (1.0, p) for p in prims])
    # print(grammar)
    # request = arrow(tinput, toutput)
    # print("Sampling with request: {}".format(request))

    # for i in range(numSamples):
    #     sample = grammar.sample(request=request)
    #     print("Program: {}".format(sample))

    return





