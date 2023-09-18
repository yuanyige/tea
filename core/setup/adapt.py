
import torch
import torch.nn as nn
from core.setup.optim import setup_optimizer
from core.setup.param import configure_model, collect_params

import core.adazoo.energy as energy
import core.adazoo.tent as tent
import core.adazoo.norm as norm
import core.adazoo.eata as eata

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
    model = configure_model(model)
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
        corruption = 'original'
        fisher_dataset, fisher_loader = eata.prepare_test_data(corruption=corruption, use_transforms=True, batch_size=cfg.OPTIM.BATCH_SIZE, im_sz=cfg.CORRUPTION.IMG_SIZE)
        fisher_dataset.set_dataset_size(cfg.EATA.FISHER_SIZE)
        fisher_dataset.switch_mode(True, False)

        model = eata.configure_model(model)
        params, param_names = eata.collect_params(model)
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
        eta_model = eata.EATA(model, optimizer, fishers, cfg.EATA.FISHER_ALPHA, e_margin=cfg.EATA.E_MARGIN, d_margin=cfg.EATA.D_MARGIN)
    else:
        eta_model = eata.EATA(model, optimizer, e_margin=cfg.EATA.E_MARGIN, d_margin=cfg.EATA.D_MARGIN)
    return eta_model

def setup_energy(model, cfg, logger):
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
                           logger = logger
                           )
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return energy_model
