import dill
import torch.nn as nn
import torch

from dreamcoder.domains.list.handwrittenProperties import handWrittenProperties, getHandwrittenPropertiesFromTemplates, tinput, toutput
from dreamcoder.domains.list.listPrimitives import basePrimitives, primitives, McCarthyPrimitives, bootstrapTarget_extra, no_length, josh_primitives
from dreamcoder.domains.list.makeListTasks import make_list_bootstrap_tasks, sortBootstrap, EASYLISTTASKS, joshTasks
from dreamcoder.domains.list.experiments.utilsPropertySampling import getPropertySamplingGrammar
from dreamcoder.properties.propertySignatureExtractor import PropertySignatureExtractor
from dreamcoder.recognition import DummyFeatureExtractor, RecognitionModel
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure
from dreamcoder.utilities import flatten, numberOfCPUs

DATA_DIR = "data/prop_sig/"
SAMPLED_PROPERTIES_DIR = "sampled_properties/"
GRAMMARS_DIR = "grammars/"
PROP_RETURN_TYPES = [tbool]

def list_options(parser):

    # parser.add_argument("--iterations", type=int, default=10)
    # parser.add_argument("--useDSL", action="store_true", default=False)
    parser.add_argument("--libraryName",  default="josh_3", choices=[
        "josh_1",
        "josh_2",
        "josh_3",
        "josh_3.1",
        "josh_final",
        "josh_rich_0_10",
        "josh_rich_0_99",
        "property_prims",
        "dc_list_domain"])
    # parser.add_argument("--propSamplingPrimitives", default="same", choices=[
    #     "same",
    #     "josh_1",
    #     "josh_2",
    #     "josh_3",
    #     "josh_3.1",
    #     "josh_final",
    #     "josh_rich_0_10",
    #     "josh_rich_0_99",
    #     "property_prims",
    #     "list_prims"])
    parser.add_argument(
        "--dataset",
        type=str,
        default="josh_3",
        choices=[
            "josh_1",
            "josh_2",
            "josh_3",
            "josh_3_long_inputs_0_10",
            "josh_3.1",
            "josh_final",
            "josh_fleet_0_99",
            "josh_fleet_10_99",
            "josh_fleet_0_10",
            "Lucas-old"])
    # parser.add_argument("--extractor", default="prop_sig", choices=[
    #     "prop_sig",
    #     "learned",
    #     "combined",
    #     "dummy"
    #     ])
    # parser.add_argument("--hidden", type=int, default=64)
    
    # Arguments related to experiments/propSimMain.py for experiments enumerating from property-fitted grammar and baselines (NOT full dreamcoder run)
    # parser.add_argument("--plotName", type=str, default=None)
    # parser.add_argument("--enumerationProxy", action="store_true", default=False)
    # parser.add_argument("--helmholtzFrontiers", type=str, default=None)

try:
    from dreamcoder.recognition import RecurrentFeatureExtractor
    class LearnedFeatureExtractor(RecurrentFeatureExtractor):
        
        H = 64
        special = None

        def tokenize(self, examples):
            def sanitize(l): return [z if z in self.lexicon else "?"
                                     for z_ in l
                                     for z in (z_ if isinstance(z_, list) else [z_])]

            tokenized = []
            for xs, y in examples:
                if isinstance(y, list):
                    y = ["LIST_START"] + y + ["LIST_END"]
                else:
                    y = [y]
                y = sanitize(y)
                if len(y) > self.maximumLength:
                    return None

                serializedInputs = []
                for xi, x in enumerate(xs):
                    if isinstance(x, list):
                        x = ["LIST_START"] + x + ["LIST_END"]
                    else:
                        x = [x]
                    if len(x) > self.maximumLength:
                        return None
                    serializedInputs.append(x)

                tokenized.append((tuple(serializedInputs), y))

            return tokenized

        def __init__(self, tasks, testingTasks=[], cuda=False, grammar=None):
            self.lexicon = set(flatten((t.examples for t in tasks + testingTasks), abort=lambda x: isinstance(
                x, str))).union({"LIST_START", "LIST_END", "?"})

            # Calculate the maximum length
            self.maximumLength = float('inf') # Believe it or not this is actually important to have here
            self.maximumLength = max(len(l)
                                     for t in tasks + testingTasks
                                     for xs, y in self.tokenize(t.examples)
                                     for l in [y] + [x for x in xs])

            self.parallelTaskOfProgram = True
            self.recomputeTasks = True

            super(
                LearnedFeatureExtractor,
                self).__init__(
                lexicon=list(
                    self.lexicon),
                tasks=tasks,
                cuda=cuda,
                H=self.H,
                bidirectional=True)
except: pass

class CombinedExtractor(nn.Module):
    special = None

    def __init__(self, 
        tasks=[],
        testingTasks=[], 
        cuda=False, 
        H=64, 
        embedSize=16,
        # What should be the timeout for trying to construct Helmholtz tasks?
        helmholtzTimeout=0.25,
        # What should be the timeout for running a Helmholtz program?
        helmholtzEvaluationTimeout=0.01,
        grammar=None):
        super(CombinedExtractor, self).__init__()

        self.propSigExtractor = PropertySignatureExtractor(tasks=tasks, testingTasks=testingTasks, H=H, embedSize=embedSize, helmholtzTimeout=helmholtzTimeout, helmholtzEvaluationTimeout=helmholtzEvaluationTimeout,
            cuda=cuda, grammar=grammar)
        self.learnedFeatureExtractor = LearnedFeatureExtractor(tasks=tasks, testingTasks=testingTasks, cuda=cuda, grammar=grammar)

        # self.propSigExtractor = PropertySignatureExtractor
        # self.learnedFeatureExtractor = LearnedFeatureExtractor

        self.outputDimensionality = H
        self.recomputeTasks = True
        self.parallelTaskOfProgram = True

        self.linear = nn.Linear(2*H, H)

    def forward(self, v, v2=None):
        pass

    def featuresOfTask(self, t):

        learnedFeatureExtractorVector = self.learnedFeatureExtractor.featuresOfTask(t)
        propSigExtractorVector = self.propSigExtractor.featuresOfTask(t)

        if learnedFeatureExtractorVector is not None and propSigExtractorVector is not None:
            return self.linear(torch.cat((learnedFeatureExtractorVector, propSigExtractorVector)))
        else:
            return None

    def featuresOfTasks(self, ts, t2=None):  # Take a task and returns [features]
        """Takes the goal first; optionally also takes the current state second"""
        return [self.featuresOfTask(t) for t in ts]

    def taskOfProgram(self, p, tp):
        return self.learnedFeatureExtractor.taskOfProgram(p=p, tp=tp)


def retrieveJSONTasks(filename, features=False):
    """
    For JSON of the form:
        {"name": str,
         "type": {"input" : bool|int|list-of-bool|list-of-int,
                  "output": bool|int|list-of-bool|list-of-int},
         "examples": [{"i": data, "o": data}]}
    """
    with open(filename, "r") as f:
        loaded = json.load(f)
    TP = {
        "bool": tbool,
        "int": tint,
        "list-of-bool": tlist(tbool),
        "list-of-int": tlist(tint),
    }
    return [Task(
        item["name"],
        arrow(TP[item["type"]["input"]], TP[item["type"]["output"]]),
        [((ex["i"],), ex["o"]) for ex in item["examples"]],
        features=None,
        cache=False,
    ) for item in loaded]

def get_tasks(dataset):
    print("Loading tasks for dataset: ", dataset)
    tasks = {
        "Lucas-old": lambda: retrieveJSONTasks("data/list_tasks.json") + sortBootstrap(),
        "bootstrap": make_list_bootstrap_tasks,
        "sorting": sortBootstrap,
        "Lucas-depth1": lambda: retrieveJSONTasks("data/list_tasks2.json")[:105],
        "Lucas-depth2": lambda: retrieveJSONTasks("data/list_tasks2.json")[:4928],
        "Lucas-depth3": lambda: retrieveJSONTasks("data/list_tasks2.json"),
        "josh_1": lambda: joshTasks("1"),
        "josh_2": lambda: joshTasks("2"),
        "josh_3": lambda: joshTasks("3"),
        "josh_3_long_inputs_0_10": lambda: joshTasks("3_long_inputs_0_10"),
        "josh_3.1": lambda: joshTasks("3.1"),
        "josh_final": lambda: joshTasks("final"),
        "josh_fleet_0_99": lambda: joshTasks("fleet_0_99"),
        "josh_fleet_0_10": lambda: joshTasks("fleet_0_10"),
        "josh_fleet_10_99": lambda: joshTasks("fleet_10_99")
    }[dataset]()

    if "josh" in dataset:
        tasks = [t for t in tasks if int(t.name[:3]) < 81 and "_1" in t.name]

    return tasks

def get_primitives(libraryName):
    primLibraries = {
             "josh_1": josh_primitives("1"),
             "josh_2": josh_primitives("2"),
             "josh_3": josh_primitives("3")[0],
             "josh_3.1": josh_primitives("3.1")[0],
             "josh_final": josh_primitives("final"),
             "josh_rich_0_10": josh_primitives("rich_0_10"),
             "josh_rich_0_99": josh_primitives("rich_0_99"),
             "property_prims": handWrittenProperties(),
             "dc_list_domain": bootstrapTarget_extra()
    }
    prims = primLibraries[libraryName]
    return prims

def get_extractor_class(extractorName):
    extractor = {
        "dummy": DummyFeatureExtractor,
        "learned": LearnedFeatureExtractor,
        "prop_sig": PropertySignatureExtractor,
        "combined": CombinedExtractor
        }[extractorName]
    return extractor

def get_extractor(tasks, baseGrammar, args):

    extractorName = args.pop("extractor")
    extractor = get_extractor_class(extractorName)
    propFilename = args["propFilename"]

    if extractorName == "learned":
        return extractor(tasks=tasks, testingTasks=[], cuda=args["cuda"], grammar=baseGrammar)

    elif extractorName == "prop_sig" or extractorName == "combined":
        if args["propToUse"] == "handwritten":
            properties = getHandwrittenPropertiesFromTemplates(tasks)
            featureExtractor = extractor(tasksToSolve=tasks, testingTasks=[], grammar=baseGrammar, cuda=False, properties=properties)
            print("Loaded {} properties from: {}".format(len(properties), "handwritten"))
        
        elif args["propToUse"] == "preloaded":
            assert propFilename is not None
            properties = dill.load(open(DATA_DIR + SAMPLED_PROPERTIES_DIR + propFilename, "rb"))
            if isinstance(properties, dict):
                assert len(properties) == 1
                properties = list(properties.values())[0]
                # filter properties that are only on inputs
                properties = [p for p in properties if "$0" in p.name]
            featureExtractor = extractor(tasksToSolve=tasks, testingTasks=[], grammar=baseGrammar, cuda=False, properties=properties)
            print("Loaded {} properties from: {}".format(len(properties), propFilename))
        
        elif args["propToUse"] == "sample":
            allProperties = {}
            tasksToSolve = tasks[0:1]

            propertyRequest = arrow(tlist(tint), tlist(tint), tbool)
            propertyGrammar = getPropertySamplingGrammar(baseGrammar, args["propSamplingGrammarWeights"], args, pseudoCounts=1, seed=args["seed"])
            try:        
                featureExtractor = extractor(tasksToSolve=tasksToSolve, testingTasks=tasks, grammar=baseGrammar, propertyGrammar=propertyGrammar, cuda=False, propertyRequest=propertyRequest)
            except AssertionError:
                raise Exception("0 properties found")

            for task in tasksToSolve:
                allProperties[task] = allProperties.get(task, []) + featureExtractor.properties
            
            for task in tasksToSolve:
                print("Found {} properties for task {}".format(len(allProperties.get(task, [])), task))
                for p in sorted(allProperties.get(task, []), key=lambda p: p.score, reverse=True):
                    print("program: {} \nreturnType: {} \nprior: {:.2f} \nscore: {:.2f}".format(p, p.request.returns(), p.logPrior, p.score))
                    print("-------------------------------------------------------------")

            if args["save"]:
                filename = "sampled_properties_weights={}_sampling_timeout={}s_seed={}.pkl".format(
                    args["propSamplingGrammarWeights"], args["propEnumerationTimeout"], args["seed"])
                savePath = DATA_DIR + SAMPLED_PROPERTIES_DIR + filename
                dill.dump(allProperties, open(savePath, "wb"))
                print("Saving sampled properties at: {}".format(savePath))
                print("Saving sampled properties at: {}".format(savePath))
                
            properties = set()
            for values in allProperties.values():
                for p in values:
                    properties.add(p)
            properties = list(properties)

        return featureExtractor

def loadSampledTasks(k=1, batchSize=100, n=10000, dslName="jrule", isSample=True):

    tasksType = "samples" if isSample else "enumerated"
    allFrontiers = []
    for i in range(0,n,batchSize):
            with open("data/prop_sig/{}_{}_{}/{}_{}-{}.pkl".format(dslName, tasksType, k, tasksType, i, i + batchSize), "rb") as f:
                frontiers = dill.load(f)
                # if sampled task has more than 11 examples keep only the first 11
                for j,f in enumerate(frontiers):
                    numExamples = min(len(f.task.examples), 11)
                    f.task.examples = f.task.examples[:numExamples]
                allFrontiers.extend(frontiers)

    # remove the 1981st frontier becuase it is too large
    if k == 1 and dslName == "josh_rich":
        allFrontiers = allFrontiers[:1981] + allFrontiers[1982:]
    return allFrontiers
