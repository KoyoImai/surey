
from torchvision import transforms, datasets

# er
from dataloaders.dataloader_er import set_loader_er_cifar10, set_valloader_er_cifar10
from dataloaders.dataloader_er import set_loader_er_cifar100, set_valloader_er_cifar100
from dataloaders.dataloader_er import set_loader_er_tinyimagenet, set_valloader_er_tinyimagenet
from dataloaders.dataloader_er import set_vanillaloader_er_cifar10, set_vanillaloader_er_cifar100, set_vanillaloader_er_tinyimagenet

# co2l
from dataloaders.dataloader_co2l import set_loader_co2l_cifar10, set_linearloader_co2l_cifar10, set_valloader_co2l_cifar10
from dataloaders.dataloader_co2l import set_loader_co2l_cifar100, set_linearloader_co2l_cifar100, set_valloader_co2l_cifar100
from dataloaders.dataloader_co2l import set_loader_co2l_tinyimagenet, set_linearloader_co2l_tinyimagenet, set_valloader_co2l_tinyimagenet

# gpm
from dataloaders.dataloader_gpm import set_loader_gpm_cifar10




def set_loader(opt, replay_indices):

    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':       # scaleから
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'tiny-imagenet':
        mean = (0.4802, 0.4480, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.mean)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    
    normalize = transforms.Normalize(mean=mean, std=std)


    # 手法毎にデータローダーの作成方法を変更する（使用するデータ拡張，データの取り出し方が異なるので）
    if opt.method == "er":
        if opt.dataset == "cifar10":
            train_loader, subset_indices = set_loader_er_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_er_cifar10(opt=opt, normalize=normalize)
            linear_loader = None
        elif opt.dataset == "cifar100":
            train_loader, subset_indices = set_loader_er_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_er_cifar100(opt=opt, normalize=normalize)
            linear_loader = None
        elif opt.dataset == "tiny-imagenet":
            train_loader, subset_indices = set_loader_er_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_er_tinyimagenet(opt=opt, normalize=normalize)
            linear_loader = None
        else:
            assert False
    
    elif opt.method == "co2l":
        if opt.dataset == "cifar10":
            train_loader, subset_indices = set_loader_co2l_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_co2l_cifar10(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_co2l_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices)
        elif opt.dataset == "cifar100":
            train_loader, subset_indices = set_loader_co2l_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_co2l_cifar100(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_co2l_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices)
        elif opt.dataset == 'tiny-imagenet':
            train_loader, subset_indices = set_loader_co2l_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_co2l_tinyimagenet(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_co2l_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)

    elif opt.method == "gpm":
        if opt.dataset == "cifar10":
            train_loader, subset_indices = set_loader_er_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_er_cifar10(opt=opt, normalize=normalize)
            linear_loader = None
        elif opt.dataset == "cifar100":
            train_loader, subset_indices = set_loader_er_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_er_cifar100(opt=opt, normalize=normalize)
            linear_loader = None
        elif opt.dataset == "tiny-imagenet":
            train_loader, subset_indices = set_loader_er_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_er_tinyimagenet(opt=opt, normalize=normalize)
            linear_loader = None
        else:
            assert False

    # データ拡張も特に加えていない現在タスクのデータローダ
    # （gpmのメモリ更新などで普通の画像が必要な手法用）
    if opt.dataset == "cifar10":
        vanilla_loader, _ = set_vanillaloader_er_cifar10(opt=opt, normalize=normalize)
    elif opt.dataset == "cifar100":
        vanilla_loader, _ = set_vanillaloader_er_cifar100(opt=opt, normalize=normalize)
    elif opt.dataset == "tiny-imagenet":
        vanilla_loader, _ = set_vanillaloader_er_tinyimagenet(opt=opt, normalize=normalize)


    dataloader = {"train": train_loader, "linear": linear_loader, "val": val_loader, "vanilla": vanilla_loader}
    return dataloader, subset_indices