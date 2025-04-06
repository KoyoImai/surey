import logging
import numpy as np

import torch
from util import AverageMeter

logger = logging.getLogger(__name__)



def train_gpm_first(opt, model, criterion, optimizer, scheduler, train_loader, epoch):

    # trainモードに変更
    model.train()

    # 学習記録
    losses = AverageMeter()

    for idx, (images, labels) in enumerate(train_loader):

        # gpu上に配置
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        
        # バッチサイズ
        bsz = labels.shape[0]

        # モデルにデータを入力して出力を取得
        y_pred = model(images)
        # print("y_pred.shape: ", y_pred.shape)

        # 損失を計算
        loss = criterion(y_pred, labels)
        # print("loss: ", loss)

        # update metric
        losses.update(loss.item(), bsz)

        # 最適化ステップ
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 学習記録の表示
        if (idx+1) % opt.print_freq == 0 or idx+1 == len(train_loader):
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), loss=losses))




def train_gpm_other(opt, model, criterion, optimizer, scheduler, train_loader, epoch, method_tools):

    # modelをtrainモードに変更
    model.train()

    feature_mat = method_tools["feature_mat"]

    # 学習記録
    losses = AverageMeter()

    for idx, (images, labels) in enumerate(train_loader):

        # gpu上に配置
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        # バッチサイズ
        bsz = labels.shape[0]

        # モデルにデータを入力して出力を取得
        y_pred = model(images)
        # print("y_pred.shape: ", y_pred.shape)

        # 損失を計算
        loss = criterion(y_pred, labels)

        # update metric
        losses.update(loss.item(), bsz)

        # 最適化ステップ
        optimizer.zero_grad()
        loss.backward()

        # 勾配の制約（勾配を問題ない方向に射影）
        kk = 0 
        for k, (m,params) in enumerate(model.named_parameters()):
            if len(params.size())==4:
                sz =  params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                    feature_mat[kk]).view(params.size())
                kk+=1
            elif len(params.size())==1 and opt.target_task !=0:
                params.grad.data.fill_(0)


        optimizer.step()

        # 学習記録の表示
        if (idx+1) % opt.print_freq == 0 or idx+1 == len(train_loader):
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), loss=losses))



    assert False









