import logging
import copy

import torch.optim.lr_scheduler as lr_scheduler

from trains.train_er import train_er, val_er, ncm_er
from trains.train_co2l import ncm_co2l, train_co2l, val_co2l
from trains.train_gpm import train_gpm_first, train_gpm_other, val_gpm, ncm_gpm
from trains.train_lucir import train_lucir, val_lucir, ncm_lucir
from trains.train_fsdgpm import train_fsdgpm, val_fsdgpm


logger = logging.getLogger(__name__)


def train(opt, model, model2, criterion, optimizer, scheduler, dataloader, epoch, method_tools):

    # データローダーの分解
    train_loader = dataloader["train"]
    val_loader = dataloader["val"]
    linear_loader = dataloader["linear"]
    vanilla_loader = dataloader["vanilla"]
    ncm_loader = dataloader["ncm"]

    
    if opt.method == "er":
        loss, _ = train_er(opt, model, model2, criterion, optimizer,
                           scheduler, train_loader, val_loader, epoch)
        classil_acc, taskil_acc = val_er(opt, model, model2, criterion,
                                         optimizer, scheduler, train_loader, val_loader, epoch)
        ncm_acc = ncm_er(model, ncm_loader, val_loader)

        logger.info(f"task {opt.target_task} Epoch {epoch}: train_loss={loss:.4f}, \
                    ClassIL_accuracy={classil_acc:.3f}, TaskIL_accuracy={taskil_acc:.3f}, NCM_accuracy={ncm_acc:.3f}")
    
    elif opt.method == "co2l":
        
        loss, model2 = train_co2l(opt=opt, model=model, model2=model2,
                                  criterion=criterion, optimizer=optimizer,
                                  scheduler=scheduler, train_loader=train_loader, epoch=epoch)
        if epoch % 50 == 0:
            classil_acc, taskil_acc = val_co2l(opt, model, model2, linear_loader, val_loader, epoch)
            ncm_acc = ncm_co2l(model, ncm_loader, val_loader)

            logger.info(f"task {opt.target_task} Epoch {epoch}: train_loss={loss:.4f}, \
                        ClassIL_accuracy={classil_acc:.3f}, TaskIL_accuracy={taskil_acc:.3f}, NCM_accuracy={ncm_acc:.3f}")
    
    elif opt.method == "gpm":

        if opt.target_task == 0:
            loss = train_gpm_first(opt, model, criterion, optimizer, scheduler, train_loader, epoch)
        else:
            loss = train_gpm_other(opt=opt, model=model, criterion=criterion, optimizer=optimizer,
                                   scheduler=scheduler, train_loader=train_loader, epoch=epoch, method_tools=method_tools)
        
        classil_acc, taskil_acc = val_gpm(opt, model, criterion, optimizer, scheduler, train_loader, val_loader, epoch)
        ncm_acc = ncm_gpm(model, ncm_loader, val_loader)

        logger.info(f"task {opt.target_task} Epoch {epoch}: train_loss={loss:.4f}, \
                    ClassIL_accuracy={classil_acc:.3f}, TaskIL_accuracy={taskil_acc:.3f}, NCM_accuracy={ncm_acc:.3f}")

    elif opt.method == "lucir":

        scheduler.step()
        cur_lamda = method_tools['cur_lamda']
        loss, model2 = train_lucir(opt, model, model2, criterion, optimizer,
                                   scheduler, train_loader, val_loader, epoch, opt.temp, cur_lamda)
        classil_acc, taskil_acc = val_lucir(opt, model, model2, criterion, optimizer,
                                            scheduler, train_loader, val_loader, epoch)
        ncm_acc = ncm_lucir(model, ncm_loader, val_loader)

        logger.info(f"task {opt.target_task} Epoch {epoch}: train_loss={loss:.4f}, \
                    ClassIL_accuracy={classil_acc:.3f}, TaskIL_accuracy={taskil_acc:.3f}, NCM_accuracy={ncm_acc:.3f}")
    
    elif opt.method == "fs-dgpm":
        
        train_fsdgpm(opt, model, model2, criterion, optimizer, scheduler, train_loader, epoch)

        if opt.earlystop:
            assert False
        
        classil_acc, taskil_acc = val_fsdgpm(opt, model, model2, criterion, optimizer, scheduler, train_loader, val_loader, epoch)
        
        # assert False


    else:
        assert False
        

    