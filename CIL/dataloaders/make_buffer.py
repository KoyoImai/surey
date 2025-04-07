





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
        from dataloaders.buffer_er import set_replay_samples_ring
        replay_indices = set_replay_samples_ring(opt, model, prev_indices=prev_indices)
    
    elif opt.method == "gpm":
        replay_indices = []
    
    return replay_indices






    