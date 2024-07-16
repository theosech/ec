from dreamcoder.likelihoodModel import UniqueTaskSignatureScore
from dreamcoder.program import Primitive
from dreamcoder.properties.property import Property
from dreamcoder.properties.utils import convertToPropertyTasks
from dreamcoder.type import tlist, tint, tbool, arrow, baseType

def _everyOutputElGtEveryInputSameIdxEl(inputList, outputList):
    endIdx = min(len(inputList), len(outputList))
    return all([inputList[i] < outputList[i] for i in range(endIdx)])

def _inputPrefixOfOutput(inputList, outputList):
    if len(inputList) > len(outputList):
        return False
    else:
        return all([inputList[i] == outputList[i] for i in range(len(inputList))])

def _inputSuffixOfOutput(inputList, outputList):
    if len(inputList) > len(outputList):
        return False
    else:
        offset = len(outputList) - len(inputList)
        return all([inputList[i] == outputList[offset + i] for i in range(len(inputList))])

tinput = baseType("input")
toutput = baseType("output")

def handWrittenProperties(grouped=False):
    """
    A list of properties to be used by PropertySignatureExtractor to create property signatures of tasks.
    These properties were hand written by solving the training tasks from the list domain official 
    experiments and thinking about what properties we could learn that make searching for the right
    programs easier for these tasks and hopefully also generalize to the test tasks.

    Ways to generate even more properties:
        2. Boolean Combination (usually AND e.g. land(input_subset_of_output, output_is_subset_of_input))
        3. Replace "all elements of list" with "any"

    Returns:
        properties: A list of Primitives which are the properties that we believe should be useful for search
    """


    noParamProperties = [
        Primitive("output_els_in_input", arrow(tinput, toutput, tbool), 
            lambda inputList: lambda outputList: all([el in inputList for el in outputList])),

        Primitive("input_els_in_output", arrow(tinput, toutput, tbool), 
            lambda inputList: lambda outputList: all([el in outputList for el in inputList])),

        Primitive("output_same_length_as_input", arrow(tinput, toutput, tbool), 
            lambda inputList: lambda outputList: len(inputList) == len(outputList)),

        Primitive("output_shorter_than_input", arrow(tinput, toutput, tbool), 
            lambda inputList: lambda outputList: len(inputList) > len(outputList)),

        Primitive("output_list_longer_than_input", arrow(tinput, toutput, tbool), 
            lambda inputList: lambda outputList: len(inputList) < len(outputList)),

        Primitive("every_output_el_gt_every_input_same_idx_el", arrow(tinput, toutput, tbool), 
            lambda inputList: lambda outputList: _everyOutputElGtEveryInputSameIdxEl(inputList, outputList)),
 
        Primitive("input_prefix_of_output", arrow(tinput, toutput, tbool), 
            lambda inputList: lambda outputList: _inputPrefixOfOutput(inputList, outputList)),

        Primitive("input_suffix_of_output", arrow(tinput, toutput, tbool), 
            lambda inputList: lambda outputList: _inputSuffixOfOutput(inputList, outputList))
        ]

    kParamProperties = [
        Primitive("all_output_els_mod_k_equals_0", arrow(tint, tinput, toutput, tbool), 
             lambda k: lambda inputList: lambda outputList: all([el % k == 0 for el in outputList]) if k > 0 else None),
        
        Primitive("all_output_els_lt_k", arrow(tinput, toutput, tint, tbool), 
             lambda k: lambda inputList: lambda outputList: all([el < k for el in outputList])),

        Primitive("output_contains_k", arrow(tinput, toutput, tint, tbool),
             lambda k: lambda inputList: lambda outputList: k in outputList),
    ]

    inputIdxParamProperties = [
        Primitive("output_contains_input_idx_i", arrow(tint, tinput, toutput, tbool),
            lambda i: lambda inputList: lambda outputList: None if i >= len(inputList) else inputList[i] in outputList),
    ]

    outputIdxParamProperties = [
        Primitive("output_list_length_n", arrow(tint, tinput, toutput, tbool), 
            lambda n: lambda inputList: lambda outputList: len(outputList) == n)
    ]

    def output_idx_i_equals_input_idx_j(i):
        def g(j):
            def f(inputList):
                def h(outputList):
                    if (j >= len(inputList) or i >= len(outputList)):
                        return None
                    else: 
                        # print(j, i)
                        # print(inputList[j], outputList[i])
                        return (inputList[j] == outputList[i])
                return h
            return f
        return g

    inputIdxOutputIdxParamProperties = [
        Primitive("output_idx_i_equals_input_idx_j", arrow(tint, tint, tinput, toutput, tbool), output_idx_i_equals_input_idx_j)
    ]

    if grouped:
        return [noParamProperties, kParamProperties, inputIdxParamProperties, outputIdxParamProperties, inputIdxOutputIdxParamProperties]
    else:
        return noParamProperties + kParamProperties + inputIdxParamProperties + outputIdxParamProperties + inputIdxOutputIdxParamProperties


def handWrittenPropertyFuncs(handWrittenPropertyPrimitives, kMin, kMax, 
    inputIdxMax, outputIdxMax):
    
    propertyFuncs = []

    noParamProperties = handWrittenPropertyPrimitives[0]
    for prop in noParamProperties:
        propertyFuncs.append(Property(program=prop.value, request=arrow(tinput, toutput, tbool), name=prop.name))

    kParamProperties = handWrittenPropertyPrimitives[1]
    for prop in kParamProperties:
        for k in range(kMin, kMax+1):
            propertyFuncs.append(Property(program=prop.value(k), request=arrow(tinput, toutput, tbool), name=prop.name.replace("_k", "_{}".format(k))))

    inputIdxParamProperties = handWrittenPropertyPrimitives[2]
    for prop in inputIdxParamProperties:
        for idx in range(inputIdxMax+1):
            propertyFuncs.append(Property(program=prop.value(idx), request=arrow(tinput, toutput, tbool), name=prop.name.replace("idx_i", "idx_{}".format(idx))))

    outputIdxParamProperties = handWrittenPropertyPrimitives[3]
    for prop in outputIdxParamProperties:
        for idx in range(outputIdxMax+1):
            propertyFuncs.append(Property(program=prop.value(idx), request=arrow(tinput, toutput, tbool), name=prop.name.replace("_n", "_{}".format(idx)),))

    inputIdxOutputIdxParamProperties = handWrittenPropertyPrimitives[4]
    for prop in inputIdxOutputIdxParamProperties:
        for inputIdx in range(inputIdxMax+1):
            for outputIdx in range(outputIdxMax+1):
                propertyFuncs.append(Property(
                    program=prop.value(outputIdx)(inputIdx),
                    request=arrow(tinput, toutput, tbool),
                    name=prop.name.replace("idx_j", "idx_{}".format(inputIdx)).replace("idx_i", "idx_{}".format(outputIdx)), 
                    ))

    return propertyFuncs


def getHandwrittenPropertiesFromTemplates(tasks, filterEquivalent=True, minIntValue=0, defaultMaxIntValue=20):

    inputTypes = {t
                  for task in tasks
                  for t in task.request.functionArguments()}
    # maps from a type to all of the inputs that we ever saw having that type
    argumentsWithType = {
        tp: [ x
              for t in tasks
              for xs,_ in t.examples
              for tpp, x in zip(t.request.functionArguments(), xs)
              if tpp == tp]
        for tp in inputTypes
    }

    maxTaskInt = min(defaultMaxIntValue, max([k for xs in argumentsWithType[tlist(tint)] for k in xs]))
    maxInputListLength = max([len(xs) for xs in argumentsWithType[tlist(tint)]])
    maxOutputListLength = maxInputListLength

    groupedProperties = handWrittenProperties(grouped=True)
    properties = handWrittenPropertyFuncs(groupedProperties, minIntValue, maxTaskInt, maxInputListLength, maxOutputListLength)

    # keep only hand written properties that help discriminate between tasks
    if filterEquivalent:
        propertyTasks = convertToPropertyTasks(tasks, propertyRequest=properties[0].request)
        scoreModel = UniqueTaskSignatureScore(tasks=propertyTasks)

        goodProperties = []
        for prop in properties:
            score = scoreModel.scoreProperty(prop, propertyTasks)
            if score > 0:
                goodProperties.append(prop)
        print("{} out of {} properties after filtering".format(len(goodProperties), len(properties)))
        return goodProperties

    return properties


