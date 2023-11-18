import os
import logging

import torch
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model


from core.eval import evaluate_ori, evaluate_ood
from core.calibration import calibration_ori
from core.config import cfg, load_cfg_fom_args
from core.utils import set_seed, set_logger
from core.model import build_model_wrn2810bn, build_model_res18bn, build_model_res50gn
from core.setada import *

logger = logging.getLogger(__name__)

def main():
    load_cfg_fom_args()
    set_seed(cfg)
    set_logger(cfg)
    device = torch.device('cuda:0')

    # configure base model
    if 'BN' in cfg.MODEL.ARCH:
        if cfg.CORRUPTION.DATASET == 'cifar10' and cfg.MODEL.ARCH == 'WRN2810_BN':
            # use robustbench
            model = 'Standard'
            base_model = load_model(model, cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions).to(device)
        elif cfg.CORRUPTION.DATASET == 'cifar100' or cfg.CORRUPTION.DATASET == 'tin200':
            base_model = build_model_wrn2810bn(cfg.CORRUPTION.NUM_CLASSES).to(device)
            ckpt = torch.load(os.path.join(cfg.CKPT_DIR ,'{}/{}.pth'.format(cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH)))
            base_model.load_state_dict(ckpt['state_dict'])
        elif cfg.CORRUPTION.DATASET == 'pacs' or cfg.CORRUPTION.DATASET == 'mnist' :
            base_model = build_model_res18bn(cfg.CORRUPTION.NUM_CLASSES).to(device)
            ckpt = torch.load(os.path.join(cfg.CKPT_DIR ,'{}/{}.pth'.format(cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH)))
            base_model.load_state_dict(ckpt['state_dict'])
        else:
            raise NotImplementedError
    elif 'GN' in cfg.MODEL.ARCH:
        group_num=int(cfg.MODEL.ARCH.split("_")[-1])
        base_model = build_model_res50gn(group_num, cfg.CORRUPTION.NUM_CLASSES).to(device)
        ckpt = torch.load(os.path.join(cfg.CKPT_DIR ,'{}/{}.pth'.format(cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH)))
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
    evaluate_ood(model, cfg, logger, device)
    evaluate_ori(model, cfg, logger, device)
    # evaluate_adv(base_model, model, cfg, logger, device)
    # calibration_ori(model, cfg, logger, device)

if __name__ == '__main__':
    main()
