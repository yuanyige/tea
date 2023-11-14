import os
import logging

import torch
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model


from core.eval import evaluate_ori, evaluate_ood
from core.calibration import calibration_ori
from core.conf import cfg, load_cfg_fom_args
from core.utils import set_seed, set_logger, build_model_res50gn, build_model_res18bn, build_model_res50in
from core.setup.adapt import *

logger = logging.getLogger(__name__)

def main(trial=None):
    load_cfg_fom_args()
    set_seed(cfg)
    set_logger(cfg)

    # cfg.OPTIM.STEPS = trial.suggest_int("step", 1, 5, step=1)
    # cfg.OPTIM.LR = trial.suggest_loguniform('lr', 1e-6, 1e-3)
    # logger.info('Optuna number of trial: '+str(trial.number))
    # logger.info('Optuna number of trial: '+str(trial.params))
    # cfg.EBM.STEPS = trial.suggest_int("sgld_step", 20, 100, step=40)
    # cfg.EBM.SGLD_LR = trial.suggest_float("sgld_lr", 0.01, 0.1, step=0.05) 

    device = torch.device('cuda:0')

    # configure base model
    if 'GN' in cfg.MODEL.ARCH:
        group_num=int(cfg.MODEL.ARCH.split("_")[-1])
        base_model = build_model_res50gn(group_num, cfg.CORRUPTION.NUM_CLASSES).to(device)
        ckpt = torch.load(os.path.join(cfg.CKPT_DIR ,'{}/ResNet50G{}.pth'.format(cfg.CORRUPTION.DATASET,group_num)))
        base_model.load_state_dict(ckpt['state_dict'])
    elif 'IN' in cfg.MODEL.ARCH:
        base_model = build_model_res50in(cfg.CORRUPTION.NUM_CLASSES).to(device)
        ckpt = torch.load(os.path.join(cfg.CKPT_DIR ,'{}/ResNet50I.pth'.format(cfg.CORRUPTION.DATASET)))
        base_model.load_state_dict(ckpt['state_dict'])
    else:
        if (cfg.CORRUPTION.DATASET == 'cifar10') or (cfg.CORRUPTION.DATASET == 'cifar100' and cfg.MODEL.ARCH != 'Standard'):
            base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions).to(device)
        elif (cfg.CORRUPTION.DATASET == 'mnist')or (cfg.CORRUPTION.DATASET == 'tin200') or (cfg.CORRUPTION.DATASET == 'cifar100' and cfg.MODEL.ARCH == 'Standard'):
            base_model = torch.load(os.path.join(cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, str(cfg.MODEL.ARCH)+'.pt')).to(device)
        elif cfg.CORRUPTION.DATASET == 'pacs':
            base_model = build_model_res18bn(cfg.CORRUPTION.NUM_CLASSES).to(device)
            ckpt = torch.load(os.path.join(cfg.CKPT_DIR ,'{}/{}.pth'.format(cfg.CORRUPTION.DATASET,cfg.MODEL.ARCH)))
            base_model.load_state_dict(ckpt['state_dict'])
        else:
            raise NotImplementedError

    # configure tta model
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "norm":
        logger.info("test-time adaptation: NORM")
        model = setup_norm(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "eta":
        logger.info("test-time adaptation: ETA")
        model = setup_eata(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "eata":
        logger.info("test-time adaptation: EATA")
        model = setup_eata(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "energy":
        logger.info("test-time adaptation: ENERGY")
        model = setup_energy(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "sar":
        logger.info("test-time adaptation: SAR")
        model = setup_sar(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "shot":
        logger.info("test-time adaptation: SHOT")
        model = setup_shot(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "pl":
        logger.info("test-time adaptation: PL")
        model = setup_pl(base_model, cfg, logger)
    else:
        raise NotImplementedError
    
    # evaluate on each severity and type of corruption in turn
    # evaluate_adv(base_model, model, cfg, logger, device)
    # evaluate_ood(model, cfg, logger, device)
    # calibration_ori(model, cfg, logger, device)
    evaluate_ori(model, cfg, logger, device)



    # if use_optuna:
    #     trial.report(ret, epoch)
    #     if trial.should_prune():
    #         raise optuna.exceptions.TrialPruned()
    

if __name__ == '__main__':
    main()

# use_optuna = False
# optuna_save_dir = '/home/yuanyige/Ladiff_nll/save_optuna_200'
# if use_optuna:
#     os.makedirs(optuna_save_dir, exist_ok=True)
#     logger = get_logger(logpath=os.path.join(optuna_save_dir, 'verbose.log'))
#     study = optuna.create_study(direction='maximize')
#     study.optimize(main, n_trials=20)
#     print('\n\nbest value',study.best_value) 
#     print('best param',study.best_params) 
# else:
    # main()