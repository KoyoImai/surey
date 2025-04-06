import logging
import numpy as np

import torch
from util import AverageMeter

logger = logging.getLogger(__name__)



def train_gpm(opt, model, criterion, optimizer, scheduler, train_loader, epoch):

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










