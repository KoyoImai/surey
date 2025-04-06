from postprocesses.postprocess_gpm import postprocess_gpm



def post_process(opt, model, dataloader):

    # データローダーの分解
    train_loader = dataloader["train"]
    val_loader = dataloader["val"]
    linear_loader = dataloader["linear"]
    vanilla_loader = dataloader["vanilla"]

    if opt.method in ["er", "co2l"]:
        return
    elif opt.method == "gpm":
        postprocess_gpm(opt, model, vanilla_loader)

    assert False