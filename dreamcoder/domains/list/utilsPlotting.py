import dill
import numpy as np
import matplotlib.pyplot as plt
import pickle

from dreamcoder.domains.list.compareProperties import compare
from dreamcoder.domains.list.utilsEval import cumulativeNumberOfTasksSolved, loadEnumerationResults

DATA_DIR = "data/prop_sig/"
ENUMERATION_RESULTS_DIR = "enumerationResults/"
SAMPLED_PROPERTIES_DIR = "sampled_properties/"
ENUMERATION_TIME = 600
UNSOLVED_LL_FOR_PLOTTING = 0
UNSOLVED_TIMES_FOR_PLOTTING = -10

def satisfiesHoldout(frontier):
    return len(frontier.entries) > 0 and frontier.task.check(frontier.topK(1).entries[0].program, timeout=1.0, leaveHoldout=False)

def plotFrontiers(modelNames, filenames=None, plotTime=False, enumerationResults=None, save=True, plotName="enumerationTimes", logTransform=False):

    if enumerationResults is None:
        if filenames is None:
            assert Exception("You must either provide the filenames of the pickled enumeration results or the results themselves")
        else:
            enumerationResults = [loadEnumerationResults(filename) for filename in filenames]

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.jet(np.linspace(0,1, len(modelNames))))

    for enumerationResult, modelName in zip(enumerationResults, modelNames):
        
        frontiers, times = enumerationResult
        numTasks = len(frontiers)

        nonEmptyFrontiers = [f for f in frontiers if len(f.entries) > 0]
        logPosteriors = sorted([-f.bestPosterior.logPosterior for f in nonEmptyFrontiers if satisfiesHoldout(f)])
        print("{}: {} / {}".format(modelName, len(logPosteriors), len(nonEmptyFrontiers)))

        solvedTasks = set(f.task for f in frontiers if satisfiesHoldout(f))
        if plotTime:
            toPlot = sorted([time for task,time in times.items() if task in solvedTasks])
            print(toPlot)
            percentTasksSolved = [i / numTasks for i in range(len(toPlot))]
            # adding point to the end so that plot lines extend all the way to the far right side of the plot
            if logTransform:
                plt.plot([np.log2(max(el, 1.0)) for el in toPlot] + [np.log2(ENUMERATION_TIME)], percentTasksSolved + [percentTasksSolved[-1]], label=modelName, alpha=0.6)
            else:
                plt.plot(toPlot + [ENUMERATION_TIME], percentTasksSolved + [percentTasksSolved[-1]], label=modelName, alpha=0.6)
        else:
            toPlot = sorted([-f.bestPosterior.logPosterior for f in frontiers if f.task in solvedTasks])
            plt.plot(toPlot, [i / numTasks for i in range(len(toPlot))], label=modelName, alpha=0.6)

    plt.ylim(bottom=0, top=1)
    if logTransform:
        plt.xlabel("Log of enumeration time (in seconds)")
    else:
        plt.xlabel("Enumeration time (in seconds)")
    plt.ylabel("Percent tasks solved")
    plt.legend()
    plt.show()
    if save:
        plt.savefig("enumerationResults/{}.png".format(plotName))
    return

def plotProxyResults(modelToLogPosteriors, plotName="enumerationProxy", save=True):

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.jet(np.linspace(0,1, len(modelToLogPosteriors.keys()))))

    for modelName, task2logPosteriors in modelToLogPosteriors.items():
        logPosteriors = sorted([-lp for _,lp in task2logPosteriors.items()])
        plt.plot(logPosteriors, [i / len(logPosteriors) for i in range(len(logPosteriors))], label=modelName, alpha=0.6)

    plt.ylim(bottom=0, top=1)
    plt.legend()
    plt.show()
    if save:
        plt.savefig("enumerationResults/{}.png".format(plotName))
    return


def plotNumSampledPropertiesVersusMdl(propertiesFilename):
    enumeratedProperties = dill.load(open(DATA_DIR + SAMPLED_PROPERTIES_DIR + propertiesFilename, "rb"))
    assert len(enumeratedProperties.keys()) == 1
    allProperties = list(enumeratedProperties.values())[0]
    logPriors = sorted([-p.logPrior for p in allProperties])
    plt.plot(np.exp(logPriors), np.arange(0,len(logPriors)))
    plt.ylabel("Number of Unique Tasks Signature properties found")
    plt.xlabel("Enumeration Time")
    plt.show()

def plotNumSampledPropertiesVersusTimeout(timeouts, propertiesFilenames, handwrittenProperties, tasks, sampledFrontiers, valuesToInt):
    numHandwrittenDiscoveredArr = []
    numTotalArr = []
    for t,filename in zip(timeouts, propertiesFilenames):
        
        enumeratedProperties = dill.load(open(DATA_DIR + SAMPLED_PROPERTIES_DIR + filename, "rb"))
        assert len(enumeratedProperties.keys()) == 1
        allProperties = enumeratedProperties[tasks[0]]
        numTotalArr.append(len(allProperties))

        equivalentSampledProperties = compare(handwrittenProperties, allProperties, tasks, sampledFrontiers, valuesToInt)
        numHandwrittenDiscoveredArr.append(len(equivalentSampledProperties))

    plt.plot(timeouts, np.array(numHandwrittenDiscoveredArr) / len(handwrittenProperties))
    plt.ylabel("Percent of Handwritten properties discovered")
    plt.ylim([0, 1])
    plt.xlabel("Property Enumeration Time (s)")
    plt.show()
    return

def _getGroundTruthProgram(task, grammar, programs=None):
    """
    Ground truth program is either provided in task.program or if it is not
    return the highest ll program in programs scoring using grammar

    Returns:
        ll (float): ll of highest scoring program under grammar
        program (Program): highest ll program
    """

    if task.program is not None:
        return  grammar.logLikelihood(task.request, task.program), task.program
    else:
        assert programs is not None
        if len(programs) == 0:
            return UNSOLVED_LL_FOR_PLOTTING, None
        bestProgram = max(programs, key=lambda p: grammar.logLikelihood(task.request, p))
        return grammar.logLikelihood(task.request, bestProgram), bestProgram

def _getTaskToAllPrograms(modelNames, enumerationFilenames):
    taskToAllPrograms = {}
    for i in range(0, len(modelNames)):
        modelName, enumerationFilename = modelNames[i], enumerationFilenames[i]
        frontiers, times = loadEnumerationResults(enumerationFilename)
        task2frontier, solvedTasks = {}, set()
        for f in frontiers:
            programsForTask = taskToAllPrograms.get(f.task, [])
            if satisfiesHoldout(f):
                taskToAllPrograms[f.task] = programsForTask + [f.topK(1).entries[0].program]
    return taskToAllPrograms

def barPlotPerTaskStatistics(modelNames, enumerationFilenames, grammarFilenames=None, plotTime=True, includeUnsolved=False):
    if includeUnsolved:
        if grammarFilenames is None:
            raise Exception("You must provided the fitted grammars if you want to include the unsolved tasks in the plot")
        else:
            grammars = [dill.load(open(filename, "rb")) for filename in grammarFilenames]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.jet(np.linspace(0,1, len(modelNames))))

    frontiers, times = loadEnumerationResults(enumerationFilenames[0])
    solvedTasks = set()
    taskToAllPrograms = _getTaskToAllPrograms(modelNames, enumerationFilenames)
    solvedByAny = set([t for t in taskToAllPrograms.keys() if len(taskToAllPrograms[t]) > 0])

    for f in frontiers:
        if satisfiesHoldout(f):
            solvedTasks.add(f.task)

    if plotTime:
        toPlot = [(task,time) if (task in solvedTasks) else (task, -10) for task,time in times.items()]
    else:
        toPlot = [(f.task, -f.topK(1).entries[0].logPosterior) if f.task in solvedTasks
        else (f.task, (-_getGroundTruthProgram(f.task, grammars[0][f.task], taskToAllPrograms.get(f.task, []))[0] if includeUnsolved else UNSOLVED_LL_FOR_PLOTTING)) 
        for f in frontiers]

    if includeUnsolved:
        pass
    else:
        toPlot = [(t,lp) for (t,lp) in toPlot if t in solvedByAny]
    
    sortedTimes = sorted(toPlot, key=lambda x: x[1])
    orderedTasks, toPlot = list(zip(*sortedTimes))
    taskNames = [t.name for t in orderedTasks]
    print(toPlot)

    # Calculate optimal width
    width = (1.0 / float(len(enumerationFilenames))) / 2.0
    ax.axes.set_xticklabels(taskNames)
    ax.bar([x_coord for x_coord in range(len(orderedTasks))], toPlot, width, label=modelNames[0], align="edge", tick_label=taskNames)

    offset = width
    for i in range(1, len(modelNames)):
        modelName, enumerationFilename = modelNames[i], enumerationFilenames[i]
        frontiers, times = loadEnumerationResults(enumerationFilename)
        task2frontier, solvedTasks = {}, set()
        for f in frontiers:
            if satisfiesHoldout(f):
                solvedTasks.add(f.task)
                task2frontier[f.task] = f

        if plotTime:
            toPlot = [times[task] if task in solvedTasks else UNSOLVED_TIMES_FOR_PLOTTING for task in orderedTasks]
        else:
            toPlot = []
            for task in orderedTasks:
                if task in solvedTasks:
                    toPlot.append(-task2frontier[task].topK(1).entries[0].logPosterior)
                else:
                    if includeUnsolved:
                        ll, program = _getGroundTruthProgram(task, grammars[i][task], programs=taskToAllPrograms.get(f.task, []))
                        toPlot.append(-ll)
                    else:
                        toPlot.append(UNSOLVED_LL_FOR_PLOTTING)

        ax.bar([x_coord + offset for x_coord in range(len(orderedTasks))], toPlot, width, label=modelName, align="edge", tick_label=taskNames)
        offset += width

    ax.set_ylabel('Negative log probability', fontsize=12)
    ax.set_xlabel('Concepts', fontsize=12)
    plt.xticks(rotation=90, fontsize=6)
    plt.legend()
    plt.show()

    return

def main():
    # modelNames, enumerationFilenames, grammarFilenames = zip(*[
    # # #     ########## Random primitive weights (seed=1) #############
    # # #     # "propsim-fitted": "propsimGrammarsHandwritten_2022-01-01_23:19:30.389021_t=600.pkl",
    # # #     # "propsimGrammarsHandwrittenEqWeight": "propsimGrammarsHandwrittenEqWeight_2022-01-01_23:36:17.413812_t=600.pkl",
    # # #     # "propsimGrammarsAutomatic_2022": "propsimGrammarsAutomatic_2022-01-02_00:10:34.733461_t=600.pkl",
    # # #     # "propsimGrammarsAutomaticEqWeight_2022-01-01_23:53:34.640085_t=600.pkl",
    # # #     # "all-fitted": "helmholtzFitted_2022-01-02_00:20:35.402688_t=600.pkl",

    # #     ########## Equal weight primitives (seed=3) #############
    #     ("PropsimFit (handwritten)", "propsimGrammarsHandwritten_2022-01-03_23:13:48.748730_t=600.pkl", "data/prop_sig/helmholtz_frontiers/josh_rich_0_10_enumerated/13742_with_josh_fleet_0_10-inputs__propSim_propToUse=handwritten_numHelmFrontiers_10000_nSim=50_weightedSim=False_onlyTrueProp=_False_taskSpecificInputs=True_compressSimilar=False_equalWprop=False_seed=3_grammars.pkl"), 
    #     ("PropsimFit (automatic)", "propsimGrammarsAutomatic_2022-01-04_00:04:41.373786_t=600.pkl", "data/prop_sig/helmholtz_frontiers/josh_rich_0_10_enumerated/13742_with_josh_fleet_0_10-inputs_propSim_propToUse=sample_numHelmFrontiers_10000_nSim=50_weightedSim=False_onlyTrueProp=_False_taskSpecificInputs=True_compressSimilar=False_equalWprop=True_seed=3_grammars.pkl"), 
    #     ("Neural", "neural_2022-01-13_19:02:55.806615_t=600.pkl", None),
    # #     # "propsimGrammarsHandwrittenEqWeight_2022-01-01_22:30:59.723239_t=600.pkl",
    # #     # "propsimGrammarsAutomaticEqWeight_2022-01-01_22:47:51.005722_t=600.pkl",
    # #     # "propsimGrammarsAutomatic_2022-01-01_23:04:27.669035_t=600.pkl",
    # #     # ("AllFit", "helmholtzFitted_2022-01-04_13:45:58.645759_t=600.pkl", "data/prop_sig/helmholtz_frontiers/josh_rich_0_10_enumerated/13742_with_josh_fleet_0_10-inputs__helmholtz_10000_grammars.pkl"),
    # #     # AllFit only on tasks-specific frontiers
    #     ("AllFit", "helmholtzFitted_2022-01-04_15:51:01.836870_t=600.pkl", "data/prop_sig/helmholtz_frontiers/josh_rich_0_10_enumerated/13742_with_josh_fleet_0_10-inputs__helmholtz_10000_grammars.pkl"),
    #     ("Uniform", "uniform_2022-01-04_14:07:30.300207_t=600.pkl", None)
    # ])

    # ############### DC list domain (DSL and dataset) ###############################
    # modelNames, enumerationFilenames, grammarFilenames = zip(*[
    #     ("PropsimFit (handwritten)", "propsimGrammarsHandwritten_2022-01-07_15:00:27.163045_t=600.pkl", "data/prop_sig/helmholtz_frontiers/dc_list_domain_enumerated/12468_with_Lucas-old-inputs__propSim_propToUse=handwritten_numHelmFrontiers_10000_nSim=50_weightedSim=False_onlyTrueProp=_False_taskSpecificInputs=True_compressSimilar=False_equalWprop=False_seed=1_grammars.pkl"),
    #     # ("PropsimFit (automatic)", "propsimGrammarsAutomatic_2022-01-07_15:31:16.539899_t=600.pkl", "data/prop_sig/helmholtz_frontiers/dc_list_domain_enumerated/12468_with_Lucas-old-inputs__propSim_propToUse=sample_numHelmFrontiers_10000_nSim=50_weightedSim=False_onlyTrueProp=_False_taskSpecificInputs=True_compressSimilar=False_equalWprop=False_seed=1_grammars.pkl"),
    #     ("AllFit", "helmholtzFitted_2022-01-07_15:58:57.746408_t=600.pkl", "data/prop_sig/helmholtz_frontiers/dc_list_domain_enumerated/12468_with_Lucas-old-inputs_helmholtz_numHelmFrontiers_10000_grammars.pkl"), 
    #     # ("Uniform", "uniform_2022-01-07_16:29:22.531278_t=600.pkl", "data/prop_sig/helmholtz_frontiers/dc_list_domain_enumerated/12468_with_Lucas-old-inputs_uniform_grammar.pkl")
    # ])

    ############### Josh Rule basic DSL and jrule dataset ###############################
    # modelNames, enumerationFilenames, grammarFilenames = zip(*[
    #     ("PropsimFit (handwritten)", "propsimGrammarsHandwritten_2022-01-13_14:46:34.015674_t=600.pkl", None),
    #     ("PropsimFit (automatic)", "propsimGrammarsAutomatic_2022-01-13_15:07:30.014210_t=600.pkl", None),
    #     ("Neural", "neuralGrammars_2022-01-13_15:29:19.016480_t=600.pkl", None),
    #     ("AllFit", "helmholtzFitted_2022-01-13_15:49:21.018771_t=600.pkl", None),
    #     ("Uniform", "uniform_2022-01-13_16:08:38.013273_t=600.pkl", None)
    # ])

    ############## DC list domain DSL and jrule dataset ###############################
    modelNames, enumerationFilenames, grammarFilenames = zip(*[
        ("PropsimFit (handwritten)", "propsimGrammarsHandwritten_2022-01-13_15:36:08.915337_t=600.pkl", None),
        ("PropsimFit (automatic)", "propsimGrammarsAutomatic_2022-01-13_15:57:42.614163_t=600.pkl", None),
        ("Neural", "neuralGrammars_2022-01-13_16:16:43.037385_t=600.pkl", None),
        ("AllFit", "helmholtzFitted_2022-01-13_16:35:54.643281_t=600.pkl", None),
        ("Uniform", "uniform_2022-01-13_16:55:59.817946_t=600.pkl", None)
    ])

    for f in enumerationFilenames:
        for i in [10,25,50,100,600]:
            print(f, i, cumulativeNumberOfTasksSolved(f, i))

    # plotFrontiers(modelNames, filenames=enumerationFilenames, plotTime=True, enumerationResults=None, save=True, plotName="enumerationTimes", logTransform=False)
    # barPlotPerTaskStatistics(modelNames, enumerationFilenames, grammarFilenames, plotTime=False, includeUnsolved=False)
    return 


# handwrittenProperties = getHandwrittenPropertiesFromTemplates(tasks)
# valuesToInt = {"allFalse":0, "allTrue":1, "mixed":2}
# timeouts = [1,5,10,30,60,120,180,300,3600]
# propertiesFilenames = ["sampled_properties_weights=fitted_sampling_timeout={}s_return_types=[bool]_seed=1.pkl".format(t) for t in timeouts]
# plotNumSampledPropertiesVersusTimeout(timeouts, propertiesFilenames, handwrittenProperties, tasks, sampledFrontiers, valuesToInt)
# plotNumSampledPropertiesVersusMdl(propertiesFilenames[-1])
