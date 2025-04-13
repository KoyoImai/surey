
import math

import torch
import torch.nn as nn


from util import AverageMeter, write_csv



def train_fsdgpm(opt, model, model2, criterion, optimizer, scheduler, train_loader, epoch):

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

        # 古いタスクのデータを取り出して
        b_images, b_labels, b_tasks = model.get_batch(images, labels, opt.target_task)
        # print("b_images.shape: ", b_images.shape)
        # print("b_labels.shape: ", b_labels.shape)
        # print("b_tasks.shape: ", b_tasks.shape)
        # print()


        # sharpnessのステップ
        fast_weights = None
        inner_sz = math.ceil(len(images) / opt.inner_steps)
        meta_losses = torch.zeros(opt.inner_steps).float()

        # print("inner_sz: ", inner_sz)
        k = 0

        for j in range(0, len(images), inner_sz):
            if j + inner_sz <= len(images):
                batch_x = images[j: j + inner_sz]
                batch_y = labels[j: j + inner_sz]
            else:
                batch_x = images[j:]
                batch_y = labels[j:]
        
            # 現在タスクのデータを用いてhsarpnessの評価を行う
            fast_weights = model.update_weight(batch_x, batch_y, opt.target_task, fast_weights, criterion)
            
            # samples for weight/lambdas update are from the current task and old tasks
            meta_losses[k] = model.meta_loss(b_images, b_labels, b_tasks, fast_weights)
            k += 1

        # Taking the gradient step
        with torch.autograd.set_detect_anomaly(True):
            optimizer.zero_grad()
            model.zero_grads()
            loss = torch.mean(meta_losses)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), model.opt.grad_clip_norm)

        if len(model.M_vec) > 0:

            # update the lambdas
            if model.opt.fsdgpm_method in ['dgpm', 'xdgpm']:
                torch.nn.utils.clip_grad_norm_(model.lambdas.parameters(), model.opt.grad_clip_norm)
                if model.opt.sharpness:
                    model.opt_lamdas.step()
                else:
                    model.opt_lamdas_step()

                for idx in range(len(model.lambdas)):
                    model.lambdas[idx] = nn.Parameter(torch.sigmoid(model.opt.tmp * model.lambdas[idx]))

            # only use updated lambdas to update weight
            if model.opt.fsdgpm_method == 'dgpm':
                model.zero_grad()
                loss = model.meta_loss(b_images, b_labels, b_tasks)  # Forward without weight perturbation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), model.opt.grad_clip_norm)

            # train on the rest of subspace spanned by GPM
            model.train_restgpm()
            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
            optimizer.step()

        else:
            optimizer.step()
        
        # update metric
        losses.update(loss.item(), bsz)

        # 学習記録の表示
        if (idx+1) % opt.print_freq == 0 or idx+1 == len(train_loader):
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), loss=losses))

        if epoch == 1:
            model.push_to_mem(images, labels, torch.tensor(opt.target_task), epoch)

    assert False