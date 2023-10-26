
import torch
import torch.nn as nn

from core.setup.data import load_dataloader
from core.setup.optim import setup_optimizer
from core.setup.param import configure_model, collect_params, collect_params_sar

import core.adazoo.energy as energy
import core.adazoo.tent as tent
import core.adazoo.norm as norm
import core.adazoo.eata as eata
import core.adazoo.sar as sar

def setup_source(model, cfg, logger):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model

def setup_norm(model, cfg, logger):
    """Set up test-time normalization adaptation.
    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.Norm(model)
    logger.info(f"model for adaptation: %s", model)
    stats, stat_names = norm.collect_stats(model)
    logger.info(f"stats for adaptation: %s", stat_names)
    return norm_model

def setup_tent(model, cfg, logger):
    """Set up tent adaptation.
    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = configure_model(model, ada_param=cfg.MODEL.ADA_PARAM)
    params, param_names = collect_params(model,
                                         ada_param=cfg.MODEL.ADA_PARAM,
                                         logger=logger)
    optimizer = setup_optimizer(params, cfg, logger)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model

def setup_eata(model, cfg, logger):
    model = configure_model(model)
    params, param_names = collect_params(model,
                                         ada_param=cfg.MODEL.ADA_PARAM,
                                         logger=logger)
    optimizer = setup_optimizer(params, cfg, logger)
    # optimizer = torch.optim.SGD(params, 0.00025, momentum=0.9)
    if cfg.EATA.USE_FISHER:
        # compute fisher informatrix
        if cfg.MODEL.ADAPTATION == 'eata':
            dataset = "-".join([cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH])
        else:
            dataset = cfg.CORRUPTION.DATASET
        _, fisher_dataset, _, fisher_loader = load_dataloader(root=cfg.DATA_DIR, dataset=dataset, batch_size=cfg.OPTIM.BATCH_SIZE, if_shuffle=False, logger=logger)
        #fisher_dataset.set_dataset_size(cfg.EATA.FISHER_SIZE)
        model = configure_model(model)
        params, param_names = collect_params(model, logger=logger)
        ewc_optimizer = torch.optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        for iter_, (images, targets) in enumerate(fisher_loader, start=1):      
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            outputs = model(images)
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(fisher_loader):
                        fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
        logger.info("compute fisher matrices finished")
        del ewc_optimizer
        print('fishers',fishers)
        eta_model = eata.EATA(model, optimizer, fishers, cfg.EATA.FISHER_ALPHA, e_margin=cfg.EATA.E_MARGIN, d_margin=cfg.EATA.D_MARGIN)
        
    else:
        print('fishers',None)
        eta_model = eata.EATA(model, optimizer, e_margin=cfg.EATA.E_MARGIN, d_margin=cfg.EATA.D_MARGIN)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return eta_model

def setup_energy(model, cfg, logger):
    """Set up energy adaptation.
    """
    model = configure_model(model, ada_param=cfg.MODEL.ADA_PARAM)
    params, param_names = collect_params(model, 
                                         ada_param=cfg.MODEL.ADA_PARAM,
                                         logger=logger)
    optimizer = setup_optimizer(params, cfg, logger)
    energy_model = energy.Energy(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC,
                           buffer_size=cfg.EBM.BUFFER_SIZE,
                           sgld_steps=cfg.EBM.STEPS,
                           sgld_lr=cfg.EBM.SGLD_LR,
                           sgld_std=cfg.EBM.SGLD_STD,
                           reinit_freq=cfg.EBM.REINIT_FREQ,
                           if_cond=cfg.EBM.UNCOND,
                           n_classes=cfg.CORRUPTION.NUM_CLASSES,
                           im_sz=cfg.CORRUPTION.IMG_SIZE, 
                           n_ch = cfg.CORRUPTION.NUM_CHANNEL,
                           path = cfg.SAVE_DIR,
                           logger = logger)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return energy_model


def setup_sar(model, cfg, logger):
    """Set up SAR adaptation.
    """
    model = configure_model(model, ada_param=cfg.MODEL.ADA_PARAM)
    params, param_names = collect_params_sar(model, logger=logger)
    optimizer = setup_optimizer(params, cfg, logger)
   
    adapt_model = sar.SAR(model, optimizer, margin_e0=cfg.SAR.MARGIN_E0)

    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    
    return adapt_model

    # batch_time = AverageMeter('Time', ':6.3f')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    # progress = ProgressMeter(
    #     len(val_loader),
    #     [batch_time, top1, top5],
    #     prefix='Test: ')
    # end = time.time()
    # for i, dl in enumerate(val_loader):
    #     images, target = dl[0], dl[1]
    #     if args.gpu is not None:
    #         images = images.cuda()
    #     if torch.cuda.is_available():
    #         target = target.cuda()
    #     output = adapt_model(images)
    #     acc1, acc5 = accuracy(output, target, topk=(1, 5))

    #     top1.update(acc1[0], images.size(0))
    #     top5.update(acc5[0], images.size(0))

    #     # measure elapsed time
    #     batch_time.update(time.time() - end)
    #     end = time.time()

    #     if i % args.print_freq == 0:
    #         progress.display(i)

    # acc1 = top1.avg
    # acc5 = top5.avg

    # logger.info(f"Result under {args.corruption}. The adaptation accuracy of SAR is top1: {acc1:.5f} and top5: {acc5:.5f}")

    # acc1s.append(top1.avg.item())
    # acc5s.append(top5.avg.item())

    # logger.info(f"acc1s are {acc1s}")
    # logger.info(f"acc5s are {acc5s}")