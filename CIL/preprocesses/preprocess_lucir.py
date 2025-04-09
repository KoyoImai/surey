
import copy
import math

import torch
import torch.nn as nn

from models.resnet_cifar_lucir import SplitCosineLinear



# 教師用モデルとして現在のモデルをコピーし，現在モデルのfc層を次のタスクに向けて拡張
def preprocess_lucir(opt, model, model2, method_tools):

    if opt.target_task == 0:
        return method_tools, model, model2
    elif opt.target_task == 1:
        
        # 教師モデルのコピー
        model2 = copy.deepcopy(model)

        # 現在の分類層の入力次元数と出力次元数を獲得（fc層を追加するため）
        in_features = model.head.in_features
        out_features = model.head.out_features

        print("in_features: ", in_features)
        print("out_features: ", out_features)

        new_fc = SplitCosineLinear(in_features, out_features, opt.cls_per_task)
        # print("new_fc.weight.data.shape: ", new_fc.weight.data.shape)

        new_fc.fc1.weight.data = model.head.weight.data
        print("new_fc.fc1.weight.data.shape: ", new_fc.fc1.weight.data.shape)

        new_fc.sigma.data = model.head.sigma.data
        print("new_fc.sigma.data.shape: ", new_fc.sigma.data.shape)

        model.head = new_fc


        lamda_mult = out_features*1.0 / opt.cls_per_task
        cur_lamda = opt.lamda * math.sqrt(lamda_mult)
        method_tools['cur_lamda': cur_lamda]

        # print("model: ", model)

        
        if torch.cuda.is_available():
            model = model.cuda()
            model2 = model2.cuda()

        return method_tools, model, model2
    else:

        assert False
    


    

