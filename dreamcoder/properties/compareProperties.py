import numpy as np

from dreamcoder.properties.propSim import getPropertySimTasksMatrix

def compare(handwritten, sampled, tasks, sampledFrontiers, valuesToInt):
    """
    Given a list of handwritten properties, and a list of sampled properties, and a set of tasks and sampled frontiers,
    compares the two lists based on their values for these tasks
    """
    handwrittenMatrix = getPropertySimTasksMatrix(tasks, handwritten, valuesToInt)
    sampledMatrix = getPropertySimTasksMatrix(tasks, sampled, valuesToInt)

    handWrittenToSampledMap = {}
    for i in range(handwrittenMatrix.shape[1]):
        mask = [np.array_equal(handwrittenMatrix[:, i], sampledMatrix[:, j]) for j in range(sampledMatrix.shape[1])]
        if any(mask):
            equivalentIdxs = np.where(mask)[0]
            equivalentProperties = [p for k,p in enumerate(sampled) if k in equivalentIdxs]
            handWrittenToSampledMap[handwritten[i]] = equivalentProperties
        else:
            handWrittenToSampledMap[handwritten[i]] = None

    for handwrittenProperty, equivalentProperties in handWrittenToSampledMap.items():
        print("\nHandwritten: {}".format(handwrittenProperty))
        print("-----------------------------------------------")
        if equivalentProperties is not None:
            for p in equivalentProperties:
                print(p)

    equivalentSampledProperties = [v[0] for v in handWrittenToSampledMap.values() if v is not None]
    print("{} out of {} properties found by sampling".
        format(len(equivalentSampledProperties), len(handwritten)))

    return equivalentSampledProperties
