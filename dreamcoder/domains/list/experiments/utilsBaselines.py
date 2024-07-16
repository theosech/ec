import dill
import numpy as np
import os
import torch

from dreamcoder.type import arrow, tlist, tint
from dreamcoder.recognition import RecognitionModel
from dreamcoder.properties.utils import createFrontiersWithInputsFromTask

ADD_P = 0.0001
DEL_P = 0.0001
LOG_ALPHABET = np.log(10+1)
MAX_INPUT_LENGTH = float("inf")

def _prefixEditDistance(got, expected):

    # all of these get precomputed at compile time 
    log_add_p = np.log(ADD_P);
    log_del_p = np.log(DEL_P);
    log_1madd_p = np.log(1.0-ADD_P);
    log_1mdel_p = np.log(1.0-DEL_P);    
    
    
    # Well we can always delete the whole thing and add on the remainder
    # we don't add log_1mdel_p again here since we can't delete past the beginning
    lp = log_del_p*len(got) + (log_add_p-LOG_ALPHABET)*len(expected) + log_1madd_p;
    
    # now as long as they are equal, we can take only down that far if we want
    # here we index over mi, the length of the string that so far is equal
    for mi in range(1, min(len(got), len(expected))):
        if (got[mi-1] == expected[mi-1]):
            lp = np.logaddexp(lp, log_del_p*(len(expected)-mi)                + log_1mdel_p + 
                                (log_add_p-LOG_ALPHABET)*(len(expected)-mi) + log_1madd_p)
        else:
            break
    return lp

def getHelmholtzGrammars(baseGrammar, helmholtzFrontiers, tasks, insideOutside=1):
    grammars = {}
    for t in tasks:
        taskFrontiers = createFrontiersWithInputsFromTask(helmholtzFrontiers, t)
        grammars[t] = baseGrammar.insideOutside(taskFrontiers)
    return grammars

def getGrammarsFromEditDistSim(tasks, baseGrammar, sampledFrontiers, nSim, weight=False, pseudoCounts=1):

    task2Grammar = {}

    for t in tasks:
        weightedFrontiers = []
        for frontier in sampledFrontiers:
            p = frontier.topK(1).entries[0].program
            try:
                f = p.evaluate([])
                similarity_score = 0.0
                for x,y_expected in t.examples:
                    y_got = t.predict(f, x)
                    similarity_score += _prefixEditDistance(y_got, y_expected)
                weightedFrontiers.append((frontier, similarity_score))
            except IndexError:
                # free variable
                pass
            except Exception as e:
                # print("Exception during evaluation:", e)
                pass

        sortedFrontiers = sorted(weightedFrontiers, key=lambda el: el[1], reverse=True)[:nSim]
        similarFrontiers, similarFrontierWeights = [el[0] for el in sortedFrontiers], [el[1] - (sortedFrontiers[-1][1] - 1.0) for el in sortedFrontiers]
        # for f, w in zip(similarFrontiers, similarFrontierWeights):
        #     print("{}: {}".format(f.topK(1).entries[0].program, w))
        task2Grammar[t] = baseGrammar.insideOutside(similarFrontiers, pseudoCounts, iterations=1, 
            frontierWeights=similarFrontierWeights if weight else None, weightByProgramPrior=False)

    return task2Grammar


def getGrammarsFromNeuralRecognizer(extractor, tasks, testingTasks, baseGrammar, sampledFrontiers, save, saveDirectory, datasetName, args, pklFile=None):

    recognizer = RecognitionModel(
    featureExtractor=extractor(tasks, grammar=baseGrammar, testingTasks=testingTasks, cuda=torch.cuda.is_available()),
    grammar=baseGrammar,
    cuda=torch.cuda.is_available(),
    contextual=False,
    previousRecognitionModel=False,
    )
    
    try:
        if pklFile is not None:
            name = pklFile
        else:
            # get name of neural recognition model
            ep, CPUs, helmholtzRatio, rs, rt = args.pop("earlyStopping"), args["CPUs"], args.pop("helmholtzRatio"), args.pop("recognitionSteps"), args.pop("recognitionTimeout")
            filename = "{}_neural_ep={}_RS={}_RT={}_hidden={}_r={}_contextual={}_0_99".format(datasetName, ep, rs, rt, args["hidden"], helmholtzRatio, args["contextual"])
            path = saveDirectory + filename
            name = "{}_recognizer.pkl".format(path)

        with open(name, 'rb') as handle:
            recognizer = dill.load(handle)
            print("Loaded {}".format(name))
       
        # change the maximum allowed length of input lists. every input list has special tokens LIST_START and LIST_END hence the +2
        trainedRecognizer.featureExtractor.maximumLength = MAX_INPUT_LENGTH + 2
        grammars = {}
        for t in tasks:
            grammars[t] = recognizer.grammarOfTask(t).untorch()
        return grammars

    except FileNotFoundError:
        print("Couldn't find: {}".format(name))
        print("Trained recognizer not found, training now ...")

        # count how many tasks can be tokenized
        excludeIdx = []
        for i,f in enumerate(sampledFrontiers):
            if recognizer.featureExtractor.featuresOfTask(f.task) is None:
                excludeIdx.append(i)
        sampledFrontiers = [f for i,f in enumerate(sampledFrontiers) if i not in excludeIdx]
        print("Can't get featuresOfTask for {} tasks. Now have {} frontiers".format(len(excludeIdx), len(sampledFrontiers)))

        trainedRecognizer = recognizer.trainRecognizer(
        frontiers = [],
        helmholtzFrontiers=sampledFrontiers,
        helmholtzRatio=helmholtzRatio,
        CPUs=CPUs,
        lrModel=False, 
        earlyStopping=ep, 
        holdout=ep,
        steps=rs,
        timeout=rt,
        defaultRequest=arrow(tlist(tint), tlist(tint)))

    grammars = {}
    for task in tasks + testingTasks:
        grammar = trainedRecognizer.grammarOfTask(task).untorch()
        print("neural grammar\n", grammar)
        grammars[task] = grammar

    if save:
        if not os.path.exists(saveDirectory):
            os.makedirs(saveDirectory)
        with open("{}_grammars.pkl".format(path), 'wb') as handle:
            print("Saved recognizer grammars at: {}".format(path))
            dill.dump(grammars, handle)
        with open("{}_recognizer.pkl".format(path), 'wb') as handle:
            dill.dump(trainedRecognizer, handle)

    return grammars

