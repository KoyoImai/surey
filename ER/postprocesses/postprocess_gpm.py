import numpy as np

import torch
import torch.nn.functional as F


## Define ResNet18 model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))


def get_representation_matrix_ResNet18(opt, model, vanilla_loader):

    act_key=list(model.act.keys())
    print("act_key: ", act_key)

    # modelをevalモードに変更
    model.eval()

    # 
    with torch.no_grad():

        for idx, (images, labels) in enumerate(vanilla_loader):

            # gpu上に配置
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            
            
            # これで，model.actに各層の活性化マップが保持される
            example_out  = model(images)

            break

    act_list =[]
    act_list.extend([model.act['conv_in'], 
        model.layer1[0].act['conv_0'], model.layer1[0].act['conv_1'], model.layer1[1].act['conv_0'], model.layer1[1].act['conv_1'],
        model.layer2[0].act['conv_0'], model.layer2[0].act['conv_1'], model.layer2[1].act['conv_0'], model.layer2[1].act['conv_1'],
        model.layer3[0].act['conv_0'], model.layer3[0].act['conv_1'], model.layer3[1].act['conv_0'], model.layer3[1].act['conv_1'],
        model.layer4[0].act['conv_0'], model.layer4[0].act['conv_1'], model.layer4[1].act['conv_0'], model.layer4[1].act['conv_1']])
    
    # print("act_list[0].shape: ", act_list[0].shape)   # act_list[0].shape:  torch.Size([500, 3, 32, 32])
    # print("act_list[1].shape: ", act_list[1].shape)   # act_list[1].shape:  torch.Size([500, 20, 32, 32])
    # assert False

    # 各層で使用するバッチのサイズ
    batch_list  = [50,50,50,50,50,50,50,50,250,250,250,500,500,500,500,500,500]
    # batch_list  = [10,10,10,10,10,10,10,10,50,50,50,100,100,100,100,100,100]

    # network arch 
    stride_list = [ 1,      1,1,1,1,      2,1,1,1,      2,1,1,1,         2,1,1,1]
    map_list    = [32,  32,32,32,32,  32,16,16,16,     16,8,8,8,         8,4,4,4]
    in_channel  = [ 3,  20,20,20,20,  20,40,40,40,  40,80,80,80,  80,160,160,160]

    pad = 1
    sc_list=[5,9,13]    # Redidual Connectionに対応する層のインデックス
    p1d = (1, 1, 1, 1)
    mat_final=[]        # GPM行列を保存するためのリスト（list containing GPM Matrices ）
    mat_list=[]
    mat_sc_list=[]

    for i in range(len(stride_list)):
        if i==0:
            ksz = 3
        else:
            ksz = 3 
        bsz=batch_list[i]
        st = stride_list[i]     
        k=0

        # 畳み込みやパッチ抽出後に得られる出力の空間的サイズを計算
        s=compute_conv_output_size(map_list[i],ksz,stride_list[i],pad)

        # [カーネルサイズ*カーネルサイズ*チャネル数，マップサイズ*マップサイズ*バッチサイズ]
        mat = np.zeros((ksz*ksz*in_channel[i],s*s*bsz))
        # print("mat.shape: ", mat.shape)

        # 各層の活性マップを取り出す
        # act = act_list[i].detach().cpu().numpy()
        act = F.pad(act_list[i], p1d, "constant", 0).detach().cpu().numpy()

        # 実際の活性マップactから特定の領域のみ取り出す
        for kk in range(bsz):
            for ii in range(s):
                for jj in range(s):
                    
                    # print("act[kk, :, st*ii:ksz+st*ii, st*jj:ksz+st*jj].shape: ", act[kk, :, st*ii:ksz+st*ii, st*jj:ksz+st*jj].shape)  # 一例: (3, 3, 3)
                    # print("act[kk, :, st*ii:ksz+st*ii, st*jj:ksz+st*jj]: ", act[kk, :, st*ii:ksz+st*ii, st*jj:ksz+st*jj])

                    # 
                    mat[:, k]=act[kk, :, st*ii:ksz+st*ii, st*jj:ksz+st*jj].reshape(-1)
                    # print("torch.tensor(mat).shape: ", torch.tensor(mat).shape)  # torch.tensor(mat).shape:  torch.Size([27, 51200])
                    # print("mat[0:9, k]: ", mat[0:9, k])

                    k +=1
        mat_list.append(mat)
        # print("torch.tensor(mat_list).shape: ", torch.tensor(mat_list).shape)  # torch.Size([1, 27, 51200])
        
        # Redidual部分（For Shortcut Connection）
        if i in sc_list:
            k=0
            s=compute_conv_output_size(map_list[i],1,stride_list[i])
            mat = np.zeros((1*1*in_channel[i],s*s*bsz))
            act = act_list[i].detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        # print("act[kk,:,st*ii:1+st*ii,st*jj:1+st*jj].shape: ", act[kk,:,st*ii:1+st*ii,st*jj:1+st*jj].shape)  # act[kk,:,st*ii:1+st*ii,st*jj:1+st*jj].shape:  (20, 1, 1)
                        mat[:,k]=act[kk, :, st*ii:1+st*ii, st*jj:1+st*jj].reshape(-1)
                        k +=1
            mat_sc_list.append(mat) 
        
    ik=0
    for i in range (len(mat_list)):
        mat_final.append(mat_list[i])
        if i in [6,10,14]:
            mat_final.append(mat_sc_list[ik])
            ik+=1

    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_final)):
        print ('Layer {} : {}'.format(i+1,mat_final[i].shape))
    print('-'*30)

    return mat_final


def postprocess_gpm(opt, model, vanilla_loader):

    # 特徴行列の取得
    mat_list = get_representation_matrix_ResNet18(opt, model, vanilla_loader)
    # print("torch.tensor(mat_list).shape: ", torch.tensor(mat_list).shape)
    assert False

    # 




