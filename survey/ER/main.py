import os
import argparse
import numpy as np
import logging


import torch
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import Subset, Dataset
import torch.optim.lr_scheduler as lr_scheduler

from dataloaders.make_buffer import set_buffer
from dataloaders.make_dataloader import set_loader
from trains.main_train import train
from preprocesses.main_preprocess import pre_process
from postprocesses.main_postprocess import post_process



def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # 手法
    parser.add_argument('--method', type=str, default="")

    # logの名前（実行毎に変えてね）
    parser.add_argument('--log_name', type=str, default="practice")


    # データセット周り
    parser.add_argument('--data_folder', type=str, default='/home/kouyou/Datasets/', help='path to custom dataset')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'tiny-imagenet', 'path'], help='dataset')


    # 最適化手法
    parser.add_argument('--learning_rate', type=float, default=0.03,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    
    # 学習条件
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=None)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=10)

    # classifierの学習条件(Co2Lなど線形分類で後から評価する手法用)
    parser.add_argument('--linear_epochs', type=int, default=100)
    parser.add_argument('--linear_lr', type=float, default=0.1)
    parser.add_argument('--linear_batch_size', type=int, default=256)

    # 継続学習的設定
    parser.add_argument('--mem_size', type=int, default=500)
    parser.add_argument('--mem_type', type=str, default="ring")

    # 手法毎のハイパラ（co2l）
    parser.add_argument('--current_temp', type=float, default=0.2)
    parser.add_argument('--past_temp', type=float, default=0.1)
    parser.add_argument('--distill_power', type=float, default=0.1)

    # その他の条件
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=777)


    opt = parser.parse_args()

    return opt


def setup_logging(opt):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),                   # コンソール出力
            logging.FileHandler(f"{opt.explog_path}/experiment.log", mode="w")  # ファイルに記録（上書きモード）
        ]
    )


def preparation(opt):

    # データセット毎にタスク数・タスク毎のクラス数を決定
    if opt.dataset == 'cifar10':
        opt.n_cls = 10
        opt.cls_per_task = 2
        opt.size = 32
    if opt.dataset == 'cifar100':
        opt.n_cls = 100
        opt.cls_per_task = 10
        opt.size = 32
    elif opt.dataset == 'tiny-imagenet':
        opt.n_cls = 200
        opt.cls_per_task = 20
        opt.size = 64
    else:
        pass

    # 総タスク数
    opt.n_task = opt.n_cls // opt.cls_per_task

    # モデルの保存，実験記録などの保存先パス
    if opt.data_folder is None:
        opt.data_folder = '~/data/'
    opt.model_path = f'./logs/{opt.log_name}/model/'
    opt.explog_path = f'./logs/{opt.log_name}/exp_log/'
    opt.mem_path = f'./logs/{opt.log_name}/mem_log/'

    # ディレクトリ作成
    if not os.path.isdir(opt.model_path):
        os.makedirs(opt.model_path)
    if not os.path.isdir(opt.explog_path):
        os.makedirs(opt.explog_path)
    if not os.path.isdir(opt.mem_path):
        os.makedirs(opt.mem_path)
    

def make_setup(opt):

    from dataloaders.make_dataloader import set_loader

    method_tools = {}

    # 手法毎にモデル構造，損失関数，最適化手法，スケジューラを作成
    if opt.method == "er":

        
        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_er import BackboneResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = BackboneResNet(name='resnet18', head='linear', feat_dim=opt.n_cls)
        model2 = None
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
    
    elif opt.method == "co2l":

        from losses.loss_co2l import SupConLoss
        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_co2l import SupConResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = SupConResNet(name='resnet18', head='mlp', feat_dim=128)
        model2 = SupConResNet(name='resnet18', head='mlp', feat_dim=128)
        criterion = SupConLoss(temperature=0.07)
        optimizer = optim.SGD(model.parameters(),
                                lr=opt.learning_rate,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    elif opt.method == "gpm":
        
        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_gpm import ResNet18
        elif opt.dataset in ["imagemet"]:
            assert False
        
        
        # model = ResNet18(nf=16, nclass=opt.n_cls)
        model = ResNet18(nf=20, nclass=opt.n_cls)
        # print("model: ", model)
        # assert False
        model2 = None
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
        
        method_tools = {"feature_list": [], "threshold": None, "feature_mat": []}

    else:
        assert False

    # gpu上に配置
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        if model2 is not None:
            model2 = model2.cuda()
    
    return model, model2, criterion, optimizer, method_tools


def make_scheduler(opt, optimizer, epochs, dataloader):

    if opt.method == "er":
        scheduler = None
    elif opt.method == "co2l":
        print("len(dataloader): ", len(dataloader))
        total_steps = epochs * len(dataloader)
        if opt.target_task == 0:
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.02, total_steps=total_steps, pct_start=0.1, anneal_strategy='cos')
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=total_steps, pct_start=0.1, anneal_strategy='cos')
    elif opt.method == "gpm":
        scheduler = None
    else:
        assert False

    return scheduler

def main():

    # コマンドライン引数の処理
    opt = parse_option()

    # logの名前
    opt.log_name = f"{opt.log_name}_{opt.method}_{opt.mem_type}{opt.mem_size}_{opt.dataset}"

    # データローダ作成の前処理
    preparation(opt)

    # loggerの設定
    setup_logging(opt=opt)
    logging.info("Experiment started")

    # modelの作成，損失関数の作成，Optimizerの作成
    model, model2, criterion, optimizer, method_tools = make_setup(opt)

    # バッファ内データのインデックス
    replay_indices = None

    # タスク毎の学習エポック数
    original_epochs = opt.epochs

    # 各タスクの学習
    for target_task in range(0, opt.n_task):

        # 現在タスクの更新
        opt.target_task = target_task
        print('Start Training current task {}'.format(opt.target_task))
        logging.info('Start Training current task {}'.format(opt.target_task))

        # リプレイバッファ内にあるデータのインデックスを獲得
        replay_indices = set_buffer(opt, model, prev_indices=replay_indices)
        print("replay_indices: ", replay_indices)

        # バッファ内データのインデックスを保存（次タスクの学習時に読み込むため）
        np.save(
          os.path.join(opt.mem_path, 'replay_indices_{target_task}.npy'.format(target_task=target_task)),
          np.array(replay_indices))
        
        # データローダーの作成（バッファ内のデータも含めて）
        dataloader, subset_indices = set_loader(opt, replay_indices)


        np.save(
          os.path.join(opt.mem_path, 'subset_indices_{target_task}.npy'.format(target_task=target_task)),
          np.array(subset_indices))


        # 訓練前にエポック数を設定（初期エポックだけエポック数を変える場合に必要）
        if target_task == 0 and opt.start_epoch is not None:
            opt.epochs = opt.start_epoch
        else:
            opt.epochs = original_epochs


        # schedulerの作成
        scheduler = make_scheduler(opt=opt, epochs=opt.epochs, optimizer=optimizer, dataloader=dataloader["train"])

        # タスク開始後の前処理（gpmなどの前処理が必要な手法のため）
        method_tools = pre_process(opt=opt, model=model, dataloader=dataloader, method_tools=method_tools)

        # 訓練を実行
        for epoch in range(1, opt.epochs+1):

            # 学習 & 検証
            train(opt=opt, model=model, model2=model2, criterion=criterion,
                  optimizer=optimizer, scheduler=scheduler, dataloader=dataloader,
                  epoch=epoch, method_tools=method_tools)
            
        # タスク終了後の後処理（gpmなどの後処理が必要な手法のため）
        method_tools = post_process(opt=opt, model=model, dataloader=dataloader, method_tools=method_tools)
            





    



    return 

if __name__ == "__main__":
    main()