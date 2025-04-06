
from preprocesses.preprocess_gpm import preprocess_gpm



def pre_process(opt, model, dataloader, method_tools):

    if opt.method in ["er", "co2l"]:
        return method_tools
    elif opt.method == "gpm":
        method_tools = preprocess_gpm(opt=opt, method_tools=method_tools)

    return method_tools