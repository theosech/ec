from dreamcoder.fragmentGrammar import FragmentGrammar
from dreamcoder.domains.list.makeListTasks import filter_task_examples
from dreamcoder.domains.list.runUtils import *
from dreamcoder.domains.list.utilsBaselines import *
from dreamcoder.domains.list.utilsEval import *
from dreamcoder.properties.propSim import getPropSimGrammars
from dreamcoder.properties.utilsPlotting import plotProxyResults
from dreamcoder.properties.utils import enumerateHelmholtzOcaml

VALUES_TO_INT = {"allFalse":0, "allTrue":1, "mixed":2}

def iterative_propsim(args, tasks, baseGrammar, properties, initSampledFrontiers, saveDir):

    #################################
    # Load sampled tasks
    #################################

    taskFittedGrammars = []
    # used only for taskOfProgram method
    featureExtractor = LearnedFeatureExtractor(tasks=tasks, testingTasks=[], cuda=args["cuda"], grammar=baseGrammar)

    for propSimIteration in range(args["propNumIters"]):
        print("\nLoading helmholtz tasks for iteration {}".format(propSimIteration))

        if propSimIteration == 0:
            tasksToSolve = tasks
            sampledFrontiers = {t: initSampledFrontiers for t in tasksToSolve}
            task2FittedGrammar = {t:baseGrammar for t in tasksToSolve}
        else:
            sampledFrontiers = {}
            for t in tasksToSolve:
                sampledFrontiers[t] = enumerateHelmholtzOcaml(tasks, task2FittedGrammar[t], args["enumerationTimeout"], args["CPUs"], featureExtractor, save=False)
                print("Enumerated {} helmholtz tasks for task {}".format(len(sampledFrontiers[t]), t))
        # use subset (numHelmFrontiers) of helmholtz tasks
        for t in tasksToSolve:
            if args["numHelmFrontiers"] is not None and args["numHelmFrontiers"] < len(sampledFrontiers[t]):
                sampledFrontiers[t] = sorted(sampledFrontiers[t], key=lambda f: f.topK(1).entries[0].logPosterior, reverse=True)
                sampledFrontiers[t] = sampledFrontiers[t][:min(len(sampledFrontiers[t]), args["numHelmFrontiers"])]

        #################################
        # Get Grammars
        #################################

        try:
            propSimFilename = "propSim_propToUse={}_numHelmFrontiers_{}_nSim={}_weightedSim={}_onlyTrueProp=_{}_taskSpecificInputs={}_compressSimilar={}_equalWprop={}_seed={}_grammars.pkl".format(
            args["propToUse"], args["numHelmFrontiers"], args["nSim"], args["weightedSim"], args["onlyUseTrueProperties"], args["taskSpecificInputs"], args["compressSimilar"], args["equalWeightProperties"], args["seed"])
            # directory = DATA_DIR + "grammars/{}_primitives/enumerated_{}:{}".format(args["libraryName"], args["hmfSeed"], args["helmholtzFrontiers"].split(":")[0])
            # directory += ":{}/".format(args["numHelmFrontiers"]) if args["numHelmFrontiers"] is not None else "/"
            path = "{}_{}".format(saveDir, propSimFilename)
            task2FittedGrammar = dill.load(open(path, "rb"))
            # we don't save this so assume 0 tasks solved
            tasksSolved = []
            print("Loaded propsim grammars from pkl file at: {}".format(path))

        except FileNotFoundError:
            print("Couldn't find pickled fitted propsim grammars at path: {}\nRegenerating".format(path))

            task2FittedGrammar, tasksSolved, _ = getPropSimGrammars(
               baseGrammar,
               tasksToSolve,
               tasks, 
               sampledFrontiers, 
               properties, 
               args["onlyUseTrueProperties"], 
               args["nSim"], 
               args["propPseudocounts"], 
               args["weightedSim"], 
               compressSimilar=args["compressSimilar"], 
               weightByProgramPrior=args["weightByProgramPrior"], 
               recomputeTasksWithTaskSpecificInputs=args["taskSpecificInputs"],
               computePriorFromTasks=args["computePriorFromTasks"], 
               filterSimilarProperties=args["filterSimilarProperties"], 
               maxFractionSame=args["maxFractionSame"], 
               valuesToInt=VALUES_TO_INT,
               propSimIteration=propSimIteration,
               weightByPropertyPrior=not args["equalWeightProperties"],
               verbose=args["verbose"])

            if args["save"]:
                if propSimIteration > 0:
                    savePath = "{}_iter={}_{}".format(saveDir, propSimIteration, propSimFilename)
                else:
                    savePath = "{}_{}".format(saveDir, propSimFilename)
                print("Saving propsim grammars at: {}".format(savePath))
                dill.dump(task2FittedGrammar, open(savePath, "wb"))

        taskFittedGrammars.append(task2FittedGrammar)
        print("\nSolved {} tasks at iteration {}".format(len(tasksSolved), propSimIteration))

        fileName = "enumerationResults/propSim_2021-06-28 19:33:34.730379_t=1800.pkl"
        frontiers, times = dill.load(open(fileName, "rb"))

        tasksToSolve = [t for t in tasksToSolve if t not in tasksSolved]
        print("{} still unsolved\n".format(len(tasksToSolve)))
        if len(tasksToSolve) == 0:
            break

    return taskFittedGrammars[0]

def main(args):
       
    print("cuda: {}".format(torch.cuda.is_available())) 
    random.seed(args["seed"])

    # Load tasks, DSL and grammar
    tasks = get_tasks(args["dataset"])
    tasks = tasks[2:3] if args["singleTask"] else tasks
    prims = get_primitives(args["libraryName"])
    baseGrammar = Grammar.uniform([p for p in prims])
    print("baseGrammar", baseGrammar)

    if "josh_rich" in args["libraryName"]:
        # now that we've loaded the primitives we can parse the ground truth program string
        for t in tasks:
            # parses program string and also executes to check that I/O matches parsed program
            t.parse_program(prims)

    # get helmholtz frontiers either by loading saved file, or by enumerating new ones
    if args["helmholtzFrontiers"] is not None: 
        dslDirectory, pklName = args["helmholtzFrontiers"].split("/")
        datasetName = pklName[:pklName.index(".pkl")]
        saveDir = "{}helmholtz_frontiers/{}/{}_".format(DATA_DIR, dslDirectory, datasetName)
        helmholtzFrontiers = loadEnumeratedTasks(filename=args["helmholtzFrontiers"], primitives=prims, numExamples=8)
    else:
        datasetName = args["dataset"]
        # this is only used for its taskOfProgram method to generate synthetic tasks
        featureExtractor = LearnedFeatureExtractor(tasks=tasks, testingTasks=[], cuda=args["cuda"], grammar=baseGrammar)
        libraryName = "{}_randomWeights_seed_{}".format(args["libraryName"], args["seed"]) if args["randomGrammarWeights"] else args["libraryName"]
        helmholtzFrontiers, saveDir = enumerateHelmholtzOcaml(tasks, baseGrammar, enumerationTimeout=60, CPUs=args["CPUs"], featureExtractor=featureExtractor, save=True, libraryName=libraryName, datasetName=datasetName)
    
    helmholtzFrontiers = helmholtzFrontiers[:args["numHelmFrontiers"]]
    
    # load/generate recognition model conditional grammar 
    neuralGrammars = getGrammarsFromNeuralRecognizer(LearnedFeatureExtractor, tasks, tasks, baseGrammar, {"hidden": args["hidden"]}, helmholtzFrontiers, args["save"], saveDir, datasetName, args)
    neuralGrammars = dill.load(open("data/prop_sig/helmholtz_frontiers/josh_rich_0_10_enumerated/13742_with_josh_fleet_0_10-inputs_neural_ep=False_RS=10000_RT=3600_hidden=64_r=0.0_contextual=False_josh_fleet_0_10_grammars.pkl", "rb"))
    neuralPropsigGrammars = dill.load(open("data/prop_sig/helmholtz_frontiers/josh_rich_0_10_enumerated/13742_with_josh_fleet_0_10-inputs_prop_sig_neural_ep=False_RS=10000_RT=3600_hidden=64_r=0.0_contextual=False_josh_fleet_0_10_grammars.pkl", "rb"))
    
    # load/generate propSim conditional grammar
    args["equalWeightProperties"] = True
    _, automaticProperties = get_extractor(tasks, baseGrammar, args)
    propsimGrammarsAutomaticEqWeight = iterative_propsim(args, tasks, baseGrammar, automaticProperties, helmholtzFrontiers, saveDir=saveDir)

    args["equalWeightProperties"] = False
    # _, automaticProperties = get_extractor(tasks, baseGrammar, args)
    propsimGrammarsAutomatic = iterative_propsim(args, tasks, baseGrammar, automaticProperties, helmholtzFrontiers, saveDir=saveDir)

    args["propToUse"] = "handwritten"
    args["equalWeightProperties"] = True
    _, handwrittenProperties = get_extractor(tasks, baseGrammar, args)
    propsimGrammarsHandwrittenEqWeight = iterative_propsim(args, tasks, baseGrammar, handwrittenProperties, helmholtzFrontiers, saveDir=saveDir)    
   
    args["propToUse"] = "handwritten"
    args["equalWeightProperties"] = False
    #_, handwrittenProperties = get_extractor(tasks, baseGrammar, args)
    propsimGrammarsHandwritten = iterative_propsim(args, tasks, baseGrammar, handwrittenProperties, helmholtzFrontiers, saveDir=saveDir)
 
    # editDistGrammars = getGrammarsFromEditDistSim(tasks, baseGrammar, sampledFrontiers, args["nSim"])
    # generate helmholtzfitted grammar
    helmholtzGrammar = getHelmholtzGrammar(baseGrammar, helmholtzFrontiers, tasks, insideOutside=1)

    grammars = [propsimGrammarsHandwritten, propsimGrammarsHandwrittenEqWeight, propsimGrammarsAutomatic, propsimGrammarsAutomaticEqWeight, helmholtzGrammar, baseGrammar]
    modelNames = ["propsimGrammarsHandwritten", "propsimGrammarsHandwrittenEqWeight", "propsimGrammarsAutomatic", "propsimGrammarsAutomaticEqWeight", "helmholtzFitted", "uniform"]

    grammars = [propsimGrammarsHandwritten, propsimGrammarsHandwrittenEqWeight, propsimGrammarsAutomaticEqWeight, propsimGrammarsAutomatic, helmholtzGrammar, baseGrammar]
    modelNames = ["propsimGrammarsHandwritten", "propsimGrammarsHandwrittenEqWeight", "propsimGrammarsAutomaticEqWeight", "propsimGrammarsAutomaticEqWeight", "helmholtzFitted", "uniform"]

    if args["enumerationProxy"]:
        modelToLogPosteriors = enumerationProxy(grammars, tasks, modelNames, verbose=True)
        plotProxyResults(modelToLogPosteriors, args["plotName"], save=True)
    else:
        enumerateFromGrammars(args, tasks, grammars, modelNames, args["save"], args["plotName"])
    return
