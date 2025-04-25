
from preprocesses.preprocess_gpm import preprocess_gpm
from preprocesses.preprocess_lucir import preprocess_lucir
from preprocesses.preprocess_fsdgpm import preprocess_fsdgpm



def pre_process(opt, model, model2,  dataloader, method_tools):

    if opt.method in ["er", "co2l"]:
        return method_tools, model, model2
    elif opt.method == "gpm":
        method_tools = preprocess_gpm(opt=opt, method_tools=method_tools)
    elif opt.method == "lucir":
        method_tools, model, model2 = preprocess_lucir(opt=opt, model=model, model2=model2, method_tools=method_tools)
        return method_tools, model, model2
    elif opt.method in ["fs-dgpm"]:
        model, method_tools = preprocess_fsdgpm(opt, model, method_tools)
        return method_tools, model, model2
    elif opt.method in ["cclis"]:
        return method_tools, model, model2
    else:
        assert False

    return method_tools, model, model2