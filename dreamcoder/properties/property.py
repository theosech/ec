"""
A program with a name, a type and potentially cached values of its evaluation on Task objects
"""
import numpy as np
import random

from dreamcoder.type import arrow, tbool

class Property:
    
    def __init__(self, program, request, name=None, logPrior=None, valueRange=None, score=None):
        self.program = program
        self.request = request
        self.name = name if name is not None else str(program)
        # only applicable if property was sampled from Grammar
        self.logPrior = logPrior
        self.cachedTaskEvaluations = {}
        self.valueRange = valueRange if valueRange is not None else ["allTrue", "mixed", "allFalse"]
        self.score = score

    def __str__(self):
        return self.name

    def updateCachedTaskEvaluations(self, task, value):
        key = task.describe()
        if key in self.cachedTaskEvaluations and (self.cachedTaskEvaluations[key] != value):
            raise Exception("Cached value for {},{} differs from new one (cached: {}, new one: {})".format(task, self.name, self.cachedTaskEvaluations[key], value))
        else:
            self.cachedTaskEvaluations[key] = value
        return

    def getValue(self, task):
        key = task.describe()
        if key in self.cachedTaskEvaluations:
            taskValue = self.cachedTaskEvaluations[key]
        else:
            taskValue = getTaskPropertyValue(self.program, task)
            self.cachedTaskEvaluations[key] = taskValue
        return taskValue

    def getTaskSignature(self, tasks):
        return [self.getValue(task) for task in tasks]

    def setPropertyValuePriors(self, propertyToPriorDistribution, valuesToInt):
        self.valuePriors = {key: propertyToPriorDistribution[valuesToInt[key]] for key in valuesToInt.keys()}
        return

    def getPropertyValuePrior(self, value):
        if self.valuePriors is None:
            raise Exception("property value priors not set")
        else:
            return self.valuePriors[value]

def getTaskPropertyValues(f, task):
    """
    Args:
        f (function): the property function
        task (Task): one of the tasks we are interesed in solving

    Returns:
        values (list): list where each value corresponds to executing f on an i/o of task.examples for i/o that
        don't result in an evaluation error

    """
    values = []
    for x,y in task.examples:
        try:
            value = task.predict(f, [x[0], y])
            values.append(value)
        except Exception as e:
            values.append(None)
    return values


def getTaskPropertyEntropy(f=None, task=None, numExamples=None, values=None, nSamples=None):
    """
    Args:
        f (function): the property function
        task (Task): one of the tasks we are interesed in solving
        numExamples (int): number of examples to use for entropy calculation
        values (list): if this is provided then f and task are ignored and we assume
        that values are the results of applying f on each of the examples of task

    Returns:
        entropy (float): The entropy of the distribution of the first numExample values of f (excluding error) across task examples
    """

    def getEntropy(values):
        valueCounts = {}
        for value in values:
            # make value hashable
            if isinstance(value, list):
                value = tuple(value)
            valueCounts[value] = valueCounts.get(value, 0.0) + 1.0

        entropy = 0.0
        for count in valueCounts.values():
            p = count / len(values)
            entropy += -(p * np.log2(p))

        return entropy


    assert (f and task) or values

    values = values if values is not None else getTaskPropertyValues(f, task)
    numNonErrorValues = len(values)

    if numExamples > numNonErrorValues:
        raise Exception("Cannot calculate entropy for f: {} and task: {} because there aren't enough non-error values ({} needed)"
            .format(f, task, numExamples))
    elif numExamples == numNonErrorValues:
        entropy = getEntropy(values)
    else:
        entropy = 0.0
        for _ in range(nSamples):
            valuesToUse = random.sample(values, numExamples)
            entropy += getEntropy(valuesToUse)
        entropy = entropy / nSamples

    return entropy


def getTaskAllSamePropertyValue(f, task):
    """
    Args:
        f (function): the property function
        task (Task): one of the tasks we are interesed in solving

    Returns:
        taskAllSameValue (bool): either "allSame" or "different" based on if f results in
        the same value for all examples
        sameValue: only applies if above is "allSame", the value of the property for each of the
        examples of this task (if above is "different" then this is None)
    """

    try:
        exampleValues = [task.predict(f,[x[0], y]) for x,y in task.examples]
        if all([value == exampleValues[0] for value in exampleValues]):
            return True, exampleValues[0]
    except Exception as e:
        pass
    return False, None


def getTaskPropertyValue(f, task):
    """
    Args:
        f (function): the property function
        task (Task): one of the tasks we are interesed in solving
    """

    taskPropertyValue = "mixed"
    exampleValues = getTaskPropertyValues(f, task)
    
    if all([value is False for value in exampleValues]):
        taskPropertyValue = "allFalse"
    elif all([value is True for value in exampleValues]):
        taskPropertyValue = "allTrue"
    else:
        # print(str(e))
        # print("Task name: {}".format(task.name))
        # print("Task example input: {}".format(task.examples[0][0][0]))
        # print("Task example output: {}".format(task.examples[0][0][1]))
        # print("Program: {}".format(program))
        taskPropertyValue = "mixed"

    return taskPropertyValue



