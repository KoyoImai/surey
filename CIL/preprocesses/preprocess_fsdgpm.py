



def preprocess_fsdgpm(opt, model):

    model.eta1 = opt.eta1

    if len(model.M_vec) > 0 and opt.fsdgpm_method in ['dgpm', 'xdgpm']:

        model.eta2 = opt.eta2
        model.define_lambda_params()
        model.update_opt_lambda(model.eta2)




    return model