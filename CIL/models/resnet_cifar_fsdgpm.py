import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import os
import os.path
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import random
import pdb
import argparse,time
import math
from copy import deepcopy

from util import seed_everything



## Define ResNet18 model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=1, bias=False)



# 外部パラメータによる順伝播用のヘルパー関数
def basicblock_forward(x, block, params, idx):
    """
    BasicBlock の関数的順伝播。
    params: 外部パラメータリス
    idx: 現在のパラメータ位置。
    順序（shortcut が不要な場合）：
      [conv1.weight, bn1.weight, bn1.bias, conv2.weight, bn2.weight, bn2.bias]
    shortcut が必要な場合はさらに [shortcut.conv.weight, shortcut.bn.weight, shortcut.bn.bias] を追加。
    """
    
    # conv1
    weight_conv1 = params[idx]
    idx += 1
    weight_bn1 = params[idx]
    idx += 1
    bias_bn1 = params[idx]
    idx += 1
    out = F.conv2d(x, weight_conv1, bias=None, stride=block.conv1.stride, padding=block.conv1.padding)
    out = F.batch_norm(out, block.bn1.running_mean, block.bn1.running_var,
                       weight=weight_bn1, bias=bias_bn1, training=True)
    out = F.relu(out)

    # conv2
    weight_conv2 = params[idx]
    idx += 1
    weight_bn2 = params[idx]
    idx += 1
    bias_bn2 = params[idx]
    idx += 1
    out = F.conv2d(out, weight_conv2, bias=None, stride=block.conv2.stride, padding=block.conv2.padding)
    out = F.batch_norm(out, block.bn2.running_mean, block.bn2.running_var,
                       weight=weight_bn2, bias=bias_bn2, training=True)
    # shortcut
    if len(block.shortcut) > 0:
        weight_short = params[idx]
        idx += 1
        weight_short_bn = params[idx]
        idx += 1
        bias_short_bn = params[idx]
        idx += 1
        shortcut = F.conv2d(x, weight_short, bias=None, stride=block.shortcut[0].stride, padding=0)
        shortcut = F.batch_norm(shortcut, block.shortcut[1].running_mean, block.shortcut[1].running_var,
                                weight=weight_short_bn, bias=bias_short_bn, training=True)
    else:
        shortcut = x
    
    out = out + shortcut
    out = F.relu(out)
    
    return out, idx



def functional_layer_forward(x, layer, params, idx):
    """
    指定された Sequential で構成された層（layer1～layer4）の各 BasicBlock について，
    外部パラメータを用いた順伝播を行う．
    """
    for block in layer:
        x, idx = basicblock_forward(x, block, params, idx)
    return x, idx


# バッチ正規化のあれこれを変更したバージョン
class BasicBlock(nn.Module):
    expansion = 1
    # expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)  # 元々のコード
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
                nn.BatchNorm2d(self.expansion * planes)
            )
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x):
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = x
        self.count +=1
        out = relu(self.bn1(self.conv1(x)))
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = out
        self.count +=1
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, nf=64, nclass=100, seed=777, opt=None, zero_init_residual=False):
        super(ResNet, self).__init__()
        
        seed_everything(seed=seed)

        self.freeze_bn = opt.freeze_bn
        self.second_order = opt.second_order
        self.opt = opt

        # eta1: 重みへのせちつどうの更新ステップサイズ
        self.eta1 = opt.eta1

        # リプレイバッファの設定
        self.memories = opt.mem_size  # mem_size of M
        self.age = 0  # total number of training samples
        self.M = []

        # setup GPM
        self.mem_batch_size = opt.mem_batch_size
        self.M_vec = []    # bases of GPM
        self.M_val = []    # eigenvalues of each basis of GPM
        self.M_task = []   # the task id of each basis, only used to analysis GPM

        # setup losses
        self.loss_ce = torch.nn.CrossEntropyLoss()

        self.in_planes = nf

        self.conv1 = conv3x3(3, nf * 1, stride=1)
        # self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.bn1 = nn.BatchNorm2d(nf * 1)


        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   # <- 追加部分
        
        # self.linear=torch.nn.ModuleList()
        # self.fc = nn.Linear(nf * 8 * block.expansion * 4, nclass, bias=False)   # 元コード
        self.fc = nn.Linear(nf * 8 * block.expansion, nclass, bias=True)
        self.act = OrderedDict()

        # パラメータ初期化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # def forward(self, x, return_feat=False):
    #     bsz = x.size(0)
    #     self.act['conv_in'] = x.view(bsz, 3, 32, 32)
    #     out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32)))) 
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = self.layer4(out)
    #     out = self.avgpool(out)
    #     feat = out
        
    #     # out = avg_pool2d(out, 2)   # 元々はこれ
    #     out = out.view(out.size(0), -1)

    #     if return_feat:
    #         return self.fc(out), feat
        
    #     return self.fc(out)

    def get_params(self):
        self.vars = []
        for p in list(self.parameters()):
            if p.requires_grad:
                self.vars.append(p)
        return self.vars

    def net_forward(self, x, vars=None, svd=False):

        if svd:
            assert False

        elif vars is not None:
            # 外部パラメータによる順伝搬
            idx = 0
            
            # conv1
            weight_conv1 = vars[idx]; idx += 1
            weight_bn1 = vars[idx]; idx += 1
            bias_bn1 = vars[idx]; idx += 1
            out = F.conv2d(x, weight_conv1, bias=None, stride=self.conv1.stride, padding=self.conv1.padding)
            out = F.batch_norm(out, self.bn1.running_mean, self.bn1.running_var,
                               weight=weight_bn1, bias=bias_bn1, training=True)
            out = F.relu(out)
            # print("out.shape: ", out.shape)
            
            # layer1
            out, idx = functional_layer_forward(out, self.layer1, vars, idx)

            # layer2
            out, idx = functional_layer_forward(out, self.layer2, vars, idx)
            
            # layer3
            out, idx = functional_layer_forward(out, self.layer3, vars, idx)
            
            # layer4
            out, idx = functional_layer_forward(out, self.layer4, vars, idx)
            
            # avgpoolとflatten
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            # print("out.shape: ", out.shape)
            
            # fc
            weight_fc = vars[idx]; idx += 1
            bias_fc = vars[idx]; idx += 1
            out_fc = F.linear(out, weight_fc, bias_fc)
            # print("out_fc.shape: ", out_fc.shape)

            return out_fc

        
        else:
            # 元のパラメータで順伝搬
            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out_flat = out.view(out.size(0), -1)
            # print("out_flat.shape: ", out_flat.shape)
            out_fc = self.fc(out_flat)
            # print("out_fc.shape: ", out_fc.shape)
        
        return out_fc


    # リプレイバッファにデータを保存
    def push_to_mem(self, batch_x, batch_y, t, epoch):
        """
            Reservoir sampling to push subsampled stream of data points to replay buffer
            """
        if epoch > 1:
            return

        batch_x = batch_x.cpu()
        batch_y = batch_y.cpu()
        t = t.cpu()

        for i in range(batch_x.shape[0]):
            self.age += 1
            if len(self.M) < self.memories:
                self.M.append([batch_x[i], batch_y[i], t])
            else:
                p = np.random.randint(0, self.age)
                if p < self.memories:
                    self.M[p] = [batch_x[i], batch_y[i], t]
        
    
    
    def get_batch(self, x, y, t):
        """
            Given the new data points, create a batch of old + new data,
            where old data is sampled from the replay buffer
            """
        
        # 新しいデータ毎のtask idを作成
        t = (torch.ones(x.shape[0]).int() * t)

        if len(self.M) > 0:
            MEM = self.M
            order = np.arange(len(MEM))
            np.random.shuffle(order)
            index = order[:min(x.shape[0], len(MEM))]

            x = x.cpu()
            y = y.cpu()

            for k, idx in enumerate(index):
                ox, oy, ot = MEM[idx]
                x = torch.cat((x, ox.unsqueeze(0)), 0)
                y = torch.cat((y, oy.unsqueeze(0)), 0)
                t = torch.cat((t, ot.unsqueeze(0)), 0)

        # handle gpus if specified
        if self.cuda:
            x = x.cuda()
            y = y.cuda()
            t = t.cuda()

        return x, y, t


    def zero_grads(self):
        
        self.zero_grad()

        if len(self.M_vec) > 0 and self.opt.fsdgpm_method in ['dgpm', 'xdgpm']:
            assert False
            self.lambdas.zero_grad()


    def meta_loss(self, x, y, tasks, fast_weights=None):
        """
            Get loss of multiple tasks tasks
            """
        outputs = self.net_forward(x, fast_weights)
        
        loss = 0.0

        for task in np.unique(tasks.data.cpu().numpy()):
            task = int(task)
            idx = torch.nonzero(tasks == task).view(-1)

            loss += self.loss_ce(outputs[idx], y[idx]) * len(idx)



        return loss/len(y)



    def take_loss(self, x, y, t, fast_weights=None, criterion=None):

        outputs = self.net_forward(x, fast_weights)
        loss = criterion(outputs, y)

        return loss



    def update_weight(self, x, y, t, fast_weights, criterion):

        loss = self.take_loss(x, y, t, fast_weights, criterion)
        if fast_weights is None:
            fast_weights = self.get_params()
        
        # print("fast_weights: ", fast_weights)

        graph_required = self.second_order
        grads = list(torch.autograd.grad(loss, fast_weights, create_graph=graph_required, retain_graph=graph_required,
                                         allow_unused=True))

        # 勾配の射影
        if len(self.M_vec) > 0:
            assert False
            grads = self.grad_projection(grads)

        # 勾配のクリッピング
        for i in range(len(grads)):
            if grads[i] is not None:
                grads[i] = torch.clamp(grads[i], min=-self.opt.grad_clip_norm, max=self.opt.grad_clip_norm)


        # 重みへの摂動を適用
        if self.opt.sharpness:
            fast_weights = list(
                map(lambda p: p[1] + p[0] * self.eta1 if p[0] is not None else p[1], zip(grads, fast_weights)))
        else:
            fast_weights = list(
                map(lambda p: p[1] - p[0] * self.eta1 if p[0] is not None else p[1], zip(grads, fast_weights)))

        return fast_weights

    

def ResNet18(nf=32, nclass=100, seed=777, opt=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], nf, nclass, seed, opt=opt)








# # GPMの公式実装に準拠したバージョン
# # 可能な限り全てのモデル構造を統一したいので，書き換える
# class BasicBlock(nn.Module):
#     expansion = 1
#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(in_planes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
#                           stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
#             )
#         self.act = OrderedDict()
#         self.count = 0

#     def forward(self, x):
#         self.count = self.count % 2 
#         self.act['conv_{}'.format(self.count)] = x
#         self.count +=1
#         out = relu(self.bn1(self.conv1(x)))
#         self.count = self.count % 2 
#         self.act['conv_{}'.format(self.count)] = out
#         self.count +=1
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = relu(out)
#         return out

# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, nf, nclass):
#         super(ResNet, self).__init__()

#         self.in_planes = nf
#         self.conv1 = conv3x3(3, nf * 1, 1)
#         self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
#         self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        
#         # self.linear=torch.nn.ModuleList()
#         self.fc = nn.Linear(nf * 8 * block.expansion * 4, nclass, bias=False)
#         self.max_num_classes = nclass
#         self.act = OrderedDict()

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         bsz = x.size(0)
#         self.act['conv_in'] = x.view(bsz, 3, 32, 32)
#         out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32)))) 
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = avg_pool2d(out, 2)
#         out = out.view(out.size(0), -1)
        
#         return self.fc(out)
#         # y=[]
#         # for t,i in self.taskcla:
#         #     y.append(self.linear[t](out))
#         # return y