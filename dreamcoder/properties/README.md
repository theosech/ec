# Properties Guide

## Sampling Properties

To sample properties for any domain, follow these steps:

### Basics ###
Use the `bin/sampleProperties.py` script to sample properties. You need to specify the domain you are working with.
```bash
python bin/sampleProperties.py --domain <domain_name>
```
Replace `<domain_name>` with the domain you are interested in, such as `jrule_list` or `text`.

### Example Commands ###

For the **jrule_list** domain:
```bash
python bin/sampleProperties.py --domain jrule_list
```

For the **text** domain:
```bash
python bin/sampleProperties.py --domain text
```

## Run DreamCoder with Properties instead of neural Recognition Model
### Basics ###
To run DreamCoder for a given domain with PropSim, a property-based task-conditional grammar (fitted on tasks with similar properties) instead of a neural recognition model.  As per usual, run `python bin/[domain].py` script to run DreamCoder for a given domain. 
<br><br>
In the `bin/[domain].py` that you would like to run, you'll need make some changes. Here is the example of how to modify bin/text.py to use properties:
1. set PropertySignatureExtractor as the featureExtractor
2. add extras with property, property_sampling and domain-specific options
3. specify the property function type

For `bin/text.py` the above 3 steps correspond to the following changes to be able to run dremcoder with properties:

```diff
try:
    import binutil
except ModuleNotFoundError:
    import bin.binutil

+ from dreamcoder.type import arrow, tlist, tcharacter, tbool     # step 3

from dreamcoder.domains.text.main import main, text_options
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs

+ from dreamcoder.properties.propertySignatureExtractor import PropertySignatureExtractor     # step 1
+ from dreamcoder.properties.utils import property_options    # step 2
+ from dreamcoder.properties.utilsPropertySampling import prop_sampling_options   # step 2


if __name__ == '__main__':

    arguments = commandlineArguments(
        recognitionTimeout=7200,
        iterations=10,
        helmholtzRatio=0.5,
        topK=2,
        maximumFrontier=5,
        structurePenalty=10.,
        a=3,
        activation="tanh",
        CPUs=numberOfCPUs(),
-       featureExtractor=LearnedFeatureExtractor,        
+       featureExtractor=PropertySignatureExtractor,        # step 1: set PropertySignatureExtractor as the featureExtractor
        pseudoCounts=30.0,
+       extras=extras)      # step 2: add extras with property, property_sampling and domain-specific options

+       def extras(parser):     # step 2
+           for f in [property_options, prop_sampling_options, text_options]:       
+               parser = f(parser)

+       arguments["propertyRequest"] = arrow(tlist(tcharacter), tlist(tcharacter), tbool)   # step 3: specify the property function type
        main(arguments)
```




To use PropSim, make sure to specify:
+ `--extractor prop_sig` and `--propSim`
+ `--propToUse sample` if you want to on the fly sample properties and use these for PropSim. Or `--propToUse preloaded` or `--propToUse handwritten` but these require domain-specific manual setup (i.e. manually specifying beforehand the picklefile of properties to use)

### Example commands ###

For the **jrule_list** domain, propsim with **handwritten** properties (which need to be specified for each domain)
```
 python bin/list.py --unconditionalEnumerationTimeout 1 --enumerationTimeout 10 --solver python --propSim --no-consolidation --propToUse handwritten --propSolver python
```
For the **jrule_list** domain, propsim with on-the-fly **sampled** properties
```
python bin/list.py --unconditionalEnumerationTimeout 1 --enumerationTimeout 10 --solver python --propSim --no-consolidation --propToUse sample --propSolver python
```

For the **text**, propsim with **sampled** properties
```
python bin/text.py --unconditionalEnumerationTimeout 180 --enumerationTimeout 10 --solver python --propSim --no-consolidation --propToUse sample --propSolver python
```
