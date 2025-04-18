





def set_buffer(opt, model, prev_indices=None):

    # 手法毎にバッファの作成方法を変更
    if opt.method == "er":
        from dataloaders.buffer_er import set_replay_samples_reservoir
        from dataloaders.buffer_er import set_replay_samples_ring

        if opt.mem_type == "reservoir":
            replay_indices = set_replay_samples_reservoir(opt, model, prev_indices=prev_indices)
        elif opt.mem_type == "ring":
            replay_indices = set_replay_samples_ring(opt, model, prev_indices=prev_indices)
        else:
            assert False


    elif opt.method == "co2l":

        if opt.mem_type == "ring":
            from dataloaders.buffer_er import set_replay_samples_ring
            replay_indices = set_replay_samples_ring(opt, model, prev_indices=prev_indices)
        elif opt.mem_type == "herding":
            from dataloaders.buffer_lucir import set_replay_samples_herding
            replay_indices = set_replay_samples_herding(opt, model, prev_indices=prev_indices)
    

    elif opt.method == "gpm":
        from dataloaders.buffer_er import set_replay_samples_reservoir
        from dataloaders.buffer_er import set_replay_samples_ring

        if opt.mem_type == "reservoir":
            replay_indices = set_replay_samples_reservoir(opt, model, prev_indices=prev_indices)
        elif opt.mem_type == "ring":
            replay_indices = set_replay_samples_ring(opt, model, prev_indices=prev_indices)
        else:
            assert False


    elif opt.method == "lucir":
        from dataloaders.buffer_er import set_replay_samples_reservoir
        from dataloaders.buffer_er import set_replay_samples_ring
        from dataloaders.buffer_lucir import set_replay_samples_herding
        if opt.mem_type == "reservoir":
            replay_indices = set_replay_samples_reservoir(opt, model, prev_indices=prev_indices)
        elif opt.mem_type == "ring":
            replay_indices = set_replay_samples_ring(opt, model, prev_indices=prev_indices)
        elif opt.mem_type == "herding":
            replay_indices = set_replay_samples_herding(opt, model, prev_indices=prev_indices)
        else:
            assert False


    elif opt.method == "fs-dgpm":
        replay_indices = []
    

    


    return replay_indices






    