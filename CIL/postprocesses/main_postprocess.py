
import numpy as np

from postprocesses.postprocess_gpm import postprocess_gpm



def post_process(opt, model, model2, dataloader, method_tools):

    # データローダーの分解
    train_loader = dataloader["train"]
    val_loader = dataloader["val"]
    linear_loader = dataloader["linear"]
    vanilla_loader = dataloader["vanilla"]

    if opt.method in ["er", "co2l"]:
        return method_tools, model2
    elif opt.method == "gpm":

        # 
        # threshold = method_tools["threshold"]
        feature_list = method_tools["feature_list"]
        threshold = np.array([0.965] * 20)

        # print("threshold: ", threshold)
        # print("feature_list: ", feature_list)

        # メモリの更新（feature_listの更新）
        feature_list = postprocess_gpm(opt=opt, model=model, vanilla_loader=vanilla_loader,
                                       feature_list=feature_list, threshold=threshold)

        method_tools["feature_list"] = feature_list
        method_tools["threshold"] = threshold
    
    elif opt.method == "lucir":
        return method_tools, model2

    else:
        assert False

    return method_tools, model2