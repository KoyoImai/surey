

from extract.extract_cclis import extract_features_cclis
from extract.extract_er import extract_features_er


def extract_features(opt, model, data_loader):

    if opt.method in ["cclis", "co2l"]:

        features, labels = extract_features_cclis(opt=opt, model=model, data_loader=data_loader)

    elif opt.method in ["er"]:

        features, labels = extract_features_er(opt=opt, model=model, data_loader=data_loader)
    
    else:
        assert False
    

    return features, labels



