import logging

import torch.optim.lr_scheduler as lr_scheduler

from trains.train_er import train_er, val_er
from trains.train_co2l import train_co2l, val_co2l
from trains.train_gpm import train_gpm


logger = logging.getLogger(__name__)


def train(opt, model, model2, criterion, optimizer, scheduler, dataloader, epoch):

    # データローダーの分解
    train_loader = dataloader["train"]
    val_loader = dataloader["val"]
    linear_loader = dataloader["linear"]
    vanilla_loader = dataloader["vanilla"]

    
    if opt.method == "er":
        loss, _ = train_er(opt, model, model2, criterion, optimizer, scheduler, train_loader, val_loader, epoch)
        classil_acc, taskil_acc = val_er(opt, model, model2, criterion, optimizer, scheduler, train_loader, val_loader, epoch)

        logger.info(f"Epoch {epoch}: train_loss={loss:.4f}, ClassIL_accuracy={classil_acc:.3f}, TaskIL_accuracy={taskil_acc:.3f}")
    
    elif opt.method == "co2l":
        
        loss, model2 = train_co2l(opt=opt, model=model, model2=model2,
                                  criterion=criterion, optimizer=optimizer,
                                  scheduler=scheduler, train_loader=train_loader, epoch=epoch)
        if epoch % 50 == 0:
            classil_acc, taskil_acc = val_co2l(opt, model, model2, linear_loader, val_loader, epoch)
    
    elif opt.method == "gpm":

        if opt.target_task == 0:
            loss = train_gpm(opt, model, criterion, optimizer, scheduler, train_loader, epoch)
        else:
            assert False

        
        

    