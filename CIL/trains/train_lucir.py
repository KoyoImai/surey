import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from util import AverageMeter


logger = logging.getLogger(__name__)




def train_lucir(opt, model, model2, criterion, optimizer, scheduler, train_loader, val_loader, epoch, T, lamda):

    # modelをtrainモードに変更
    model.train()

    # model2をevalモードに変更
    model2.eval()

    # 学習記録
    losses = AverageMeter()
    correct = 0
    total = 0

    # 過去クラスの数
    num_old_classes = model2.head.out_features

    for idx, (images, labels) in enumerate(train_loader):

        # gpu上に配置
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        
        # モデルにデータを入力
        y_pred, cur_features = model(images, return_feat=True)

        # バッチサイズ
        bsz = labels.shape[0]

        if opt.target_task == 0:
            loss = criterion(y_pred, labels)
        else:

            # 過去モデルにデータを入力
            ref_outputs, ref_features = model2(images, return_feat=True)

            loss1 = nn.CosineEmbeddingLoss()(cur_features, ref_features.detach(), \
                    torch.ones(images.shape[0]).cuda()) * lamda
            loss2 = nn.CrossEntropyLoss(y_pred, labels)

            assert False
        

        # update metric
        losses.update(loss.item(), bsz)

        # 最適化ステップ
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = y_pred.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 学習記録の表示
        if (idx+1) % opt.print_freq == 0 or idx+1 == len(train_loader):
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t' \
                  'acc {correct:.3f}'.format(
                   epoch, idx + 1, len(train_loader), loss=losses, correct=100.*correct/total))
    
    return losses.avg, model2



def val_lucir(opt, model, model2, criterion, optimizer, scheduler, train_loader, val_loader, epoch):

    
    # modelをevalモードに変更
    model.eval()

    # タスク毎の精度を保持
    corr = [0.] * (opt.target_task + 1) * opt.cls_per_task
    cnt  = [0.] * (opt.target_task + 1) * opt.cls_per_task
    correct_task = 0.0

    losses = AverageMeter()

    with torch.no_grad():

        for idx, (images, labels) in enumerate(val_loader):

            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            y_pred = model(images)
            loss = criterion(y_pred, labels)

            losses.update(loss.item(), bsz)

            cls_list = np.unique(labels.cpu())
            correct_all = (y_pred.argmax(1) == labels)

            for tc in cls_list:
                mask = labels == tc
                correct_task += (y_pred[mask, (tc // opt.cls_per_task) * opt.cls_per_task : ((tc // opt.cls_per_task)+1) * opt.cls_per_task].argmax(1) == (tc % opt.cls_per_task)).float().sum()

            for c in cls_list:
                mask = labels == c
                corr[c] += correct_all[mask].float().sum().item()
                cnt[c] += mask.float().sum().item()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Acc@1 {top1:.3f} {task_il:.3f}'.format(
                          idx, len(val_loader),top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100.
                      ))
    print(' * Acc@1 {top1:.3f} {task_il:.3f}'.format(top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100.))
    
            
    classil_acc = np.sum(corr)/np.sum(cnt)*100.
    taskil_acc = correct_task/np.sum(cnt)*100.
    return classil_acc, taskil_acc


