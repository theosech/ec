try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.properties.utils import property_options
from dreamcoder.properties.utilsPropertySampling import prop_sampling_options
from dreamcoder.properties.sampleProperties import main
from dreamcoder.type import arrow, tint, tlist, tbool, tcharacter

# jrule list domain
from dreamcoder.domains.list.experiments.runUtils import list_options
from dreamcoder.domains.list.experiments.utilsPropertySampling import getPropertyGrammar

# text domain
from dreamcoder.domains.text.main import getTextGrammar, text_options
from dreamcoder.domains.text.makeTextTasks import makeTasks

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("--domain", type=str, required=True, choices=["jrule_list", "text"], help="Specify the domain to use.")
    prop_sampling_options(parser)
    property_options(parser)
    args = vars(parser.parse_args())
    
    domain = args.pop("domain")
    if domain == "jrule_list":
        list_options(parser)
        args = vars(parser.parse_args())
        grammar, tasks = getPropertyGrammar(args)
        propertyRequest = arrow(tlist(tint), tlist(tint), tbool)

    elif domain == "text":
        text_options(parser)
        args = vars(parser.parse_args())
        tasks = makeTasks()
        tasks = [t for t in tasks if t.request == arrow(tlist(tcharacter), tlist(tcharacter))]
        grammar = getTextGrammar(tasks)
        propertyRequest = arrow(tlist(tcharacter), tlist(tcharacter), tbool)
    
    main(grammar, tasks, propertyRequest, args)
