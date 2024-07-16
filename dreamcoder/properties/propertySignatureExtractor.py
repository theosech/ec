
from dreamcoder.domains.list.handwrittenProperties import getHandwrittenPropertiesFromTemplates
from dreamcoder.properties.utils import convertToPropertyTasks
from dreamcoder.properties.utilsPropertySampling import enumerateProperties

from dreamcoder.dreaming import backgroundHelmholtzEnumeration
from dreamcoder.grammar import Grammar
from dreamcoder.program import *
from dreamcoder.task import Task
from dreamcoder.type import tint, tlist

import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class PropertySignatureExtractor(nn.Module):
    
    special = None
    
    def __init__(self, 
        tasksToSolve=[],
        testingTasks=None,
        allFrontiers=None,
        cuda=False, 
        H=64,
        embedSize=16,
        # What should be the timeout for trying to construct Helmholtz tasks?
        helmholtzTimeout=0.25,
        # What should be the timeout for running a Helmholtz program?
        helmholtzEvaluationTimeout=0.01,
        propUseEmbeddings=None,
        propToUse=None,
        propScoringMethod=None,
        propSolver=None,
        propCPUs=None,
        propEnumerationTimeout=None,
        propFilename=None,
        propertyGrammar=None,
        propAddZeroToNinePrims=None,
        propUseConjunction=None,
        propertyRequest=None,
        grammar=None,
        properties=None
        ):
        super(PropertySignatureExtractor, self).__init__()

        self.special = "unique"
        self.CUDA = cuda
        self.recomputeTasks = True
        self.outputDimensionality = H
        self.useEmbeddings = propUseEmbeddings
        self.grammar = grammar
        self.propertyGrammar = propertyGrammar
        self.propAddZeroToNinePrims = propAddZeroToNinePrims
        self.propUseConjunction = propUseConjunction
        self.propScoringMethod = propScoringMethod
        self.propToUse = propToUse
        print("propToUse: {}".format(self.propToUse))
        self.propSolver = propSolver
        self.propCPUs = propCPUs
        self.propEnumerationTimeout = propEnumerationTimeout
        self.propFilename = propFilename
        self.allTasks = tasksToSolve + testingTasks
        self.tasksToSolve = tasksToSolve
        self.propertyRequest = propertyRequest
        self.propertyAllTasks = convertToPropertyTasks(self.allTasks, self.propertyRequest)
        self.propertyTasksToSolve = convertToPropertyTasks(self.tasksToSolve, self.propertyRequest)
        print("Finished creating propertyTasks")

        if self.useEmbeddings:
            self.embedding = nn.Embedding(3, embedSize)
            self.embedSize = embedSize
        else:
            self.embedSize = 1

        # maps from a requesting type to all of the inputs that we ever saw with that request
        self.requestToInputs = {
            tp: [list(map(lambda ex: ex[0][0], t.examples)) for t in self.allTasks if t.request == tp ]
            for tp in {t.request for t in self.allTasks}
        }

        inputTypes = {t
                      for task in self.allTasks
                      for t in task.request.functionArguments()}
        # maps from a type to all of the inputs that we ever saw having that type
        self.argumentsWithType = {
            tp: [ x
                  for t in self.allTasks
                  for xs,_ in t.examples
                  for tpp, x in zip(t.request.functionArguments(), xs)
                  if tpp == tp]
            for tp in inputTypes
        }

        self.requestToNumberOfExamples = {
            tp: [ len(t.examples)
                  for t in self.allTasks if t.request == tp ]
            for tp in {t.request for t in self.allTasks}
        }
        self.helmholtzTimeout = helmholtzTimeout
        self.helmholtzEvaluationTimeout = helmholtzEvaluationTimeout
        self.parallelTaskOfProgram = True

        if cuda:
            self.CUDA=True
            self.cuda()  # I think this should work?
        self.device = torch.device("cuda") if cuda else torch.device("cpu")

        newProperties = self._getProperties()
        self.properties = newProperties if properties is None else newProperties + properties
        assert len(self.properties) > 0

        self.linear = nn.Linear(len(self.properties) * self.embedSize, H)
        self.hidden = nn.Linear(H, H)
    
    def _getHelmholtzTasks(self, numHelmholtzTasks):
        """

        Returns:
            dreamTasks (list(Task)): python list of helmholtz-sampled Task objects
        """

        helmholtzFrontiers = backgroundHelmholtzEnumeration(self.tasksToSolve, self.grammar, 3,
                                                            evaluationTimeout=0.001,
                                                            special="unique")
        frontiers = helmholtzFrontiers()
        random.shuffle(frontiers)
        programs = [frontier.entries[0].program for frontier in frontiers]

        dreamtTasks = []
        i = 0
        while len(dreamtTasks) < numHelmholtzTasks or i >= len(programs):
            task = self.taskOfProgram(programs[i], self.tasksToSolve[0].request)
            if task is not None:
                dreamtTasks.append(task)
                print("program: {}".format(programs[i]))
                print("{} -> {}".format(task.examples[0][0], task.examples[0][1]))
            i += 1
        return dreamtTasks

    def _getPropertyGrammar(self):
        medianLL = median(list(self.grammar.expression2likelihood.values()))
        maxLL = max(list(self.grammar.expression2likelihood.values()))
        # if it is 0 it means the grammar is uniform
        maxLL = maxLL if maxLL < 0 else 3

        propertyPrimitives = self.grammar.primitives
        # if tinput in self.propertyRequest.functionArguments():
        #     toutputToList = Primitive("toutput_to_tlist", arrow(toutput, tlist(tint)), lambda x: x)
        #     tinputToList = Primitive("tinput_to_tlist", arrow(tinput, tlist(tint)), lambda x: x)
        #     propertyPrimitives = propertyPrimitives + [tinputToList, toutputToList]

        if self.propAddZeroToNinePrims:

            for i in range(10):
                if str(i) not in [getattr(primitive, "name", "invented_primitive") for primitive in propertyPrimitives]:
                    propertyPrimitives.append(Primitive(str(i), tint, i))

        if self.propUseConjunction:
            propertyPrimitives.append(Primitive("and", arrow(tbool, tbool, tbool), lambda a: lambda b: a and b))

        productions = [(self.grammar.expression2likelihood.get(p, maxLL), p) for p in propertyPrimitives]
        propertyGrammar = Grammar.fromProductions(productions, logVariable=maxLL)
        print("property grammar: {}".format(propertyGrammar))
        return propertyGrammar


    def _getProperties(self):

        if self.propToUse == "handwritten":
            # raise NotImplementedError
            properties = getHandwrittenPropertiesFromTemplates(self.allTasks)
            print("Loaded {} properties from: {}".format(len(properties), "handwritten"))
            return properties
        
        elif self.propToUse == "preloaded":
            raise NotImplementedError
            # assert propFilename is not None
            # properties = dill.load(open(DATA_DIR + SAMPLED_PROPERTIES_DIR + self.propFilename, "rb"))
            # if isinstance(properties, dict):
            #     assert len(properties) == 1
            #     properties = list(properties.values())[0]
            #     # filter properties that are only on inputs
            #     properties = [p for p in properties if "$0" in p.name]
            # return properties

        
        elif self.propToUse == "sample":
            self.propertyGrammar = self.propertyGrammar if self.propertyGrammar is not None else self._getPropertyGrammar()
            properties, likelihoodModel = enumerateProperties(self.propertyGrammar, self.propertyTasksToSolve, self.propertyRequest,  self.propScoringMethod, self.propSolver, self.propCPUs, self.propEnumerationTimeout, allTasks=self.propertyAllTasks)
            print("Loaded {} properties by enumerating for {}s".format(len(properties), self.propEnumerationTimeout))
            for p in properties:
                print("Property:{} ({})".format(p.name, p.score))
            return properties

        else:
            print("self.propToUse: {}".format(self.propToUse))
            raise NotImplementedError


    def forward(self, v, v2=None):

        v = torch.tanh(self.linear(v))
        v = torch.tanh(self.hidden(v))
        output = v.view(-1)
        return output

    def featuresOfTask(self, t, onlyUseTrueProperties=False):

        if onlyUseTrueProperties:
            taskPropertyValueToInt = {"allFalse":0, "allTrue":1, "mixed":0}
        else:
            taskPropertyValueToInt = {"allFalse":0, "allTrue":1, "mixed":2}

        booleanPropertyValues = []
        for prop in self.properties:
            propertyValue = prop.getValue(t)
            booleanPropertyValues.append(taskPropertyValueToInt[propertyValue])
        
        booleanPropSig = torch.tensor(booleanPropertyValues, device=self.device)
        self.booleanPropSig = booleanPropSig

        if self.useEmbeddings:
            embeddedPropSig = self.embedding(booleanPropSig).flatten()
            return self(embeddedPropSig)
        else:
            return self(booleanPropSig.float())


    def featuresOfTasks(self, ts, t2=None):  # Take a task and returns [features]
        """Takes the goal first; optionally also takes the current state second"""
        return [self.featuresOfTask(t) for t in ts]

    def taskOfProgram(self, p, tp):
        # half of the time we randomly mix together inputs
        # this gives better generalization on held out tasks
        # the other half of the time we train on sets of inputs in the training data
        # this gives better generalization on unsolved training tasks

        if random.random() < 0.5:
            def randomInput(t): return random.choice(self.argumentsWithType[t])
            # Loop over the inputs in a random order and pick the first ones that
            # doesn't generate an exception

            startTime = time.time()
            examples = []
            while True:
                # TIMEOUT! this must not be a very good program
                if time.time() - startTime > self.helmholtzTimeout: return None

                # Grab some random inputs
                xs = [randomInput(t) for t in tp.functionArguments()]
                try:
                    y = runWithTimeout(lambda: p.runWithArguments(xs), self.helmholtzEvaluationTimeout)
                    # print("Output y from below program: {}".format(y))
                    examples.append((tuple(xs),y))
                    # we want minimum 3 examples for each task
                    if len(examples) >= max(3, random.choice(self.requestToNumberOfExamples[tp])):
                        task = Task("Helmholtz", tp, examples)
                        return task
                except Exception as e:
                    # print(e)
                    #print("failed to apply program: {} \n to input: {}".format(p, xs))
                    continue

        else:
            candidateInputs = list(self.requestToInputs[tp])
            random.shuffle(candidateInputs)
            for xss in candidateInputs:
                ys = []
                for xs in xss:
                    try: y = runWithTimeout(lambda: p.runWithArguments(xs), self.helmholtzEvaluationTimeout)
                    except: break
                    ys.append(y)
                if len(ys) == len(xss):
                    return Task("Helmholtz", tp, list(zip(xss, ys)))
            return None




